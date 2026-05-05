# KGxLM.disease2gene.candidate.no345.py
import pandas as pd
import openai
import json
import time
import pickle
import re
import numpy as np
import torch
import igraph as ig
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

# 路徑設定
PrimeKGwE_entity_embedding_path = Path("HAKE_model/PrimeKGwE/entity_embedding.npy")
PrimeKGwE_relation_embedding_path = Path("HAKE_model/PrimeKGwE/relation_embedding.npy")
PrimeKG_entity_embedding_path = Path("HAKE_model/PrimeKG/entity_embedding.npy")
PrimeKG_relation_embedding_path = Path("HAKE_model/PrimeKG/relation_embedding.npy")
relation_dict_path = Path("data/PrimeKG/relations.dict")
entity_dict = "data/PrimeKG/entities.dict"
entities_type_dict = "data/PrimeKG/entities.dict.type"

# 載入
PrimeKGwE_entity_embed = np.load(PrimeKGwE_entity_embedding_path)
PrimeKGwE_relation_embed = np.load(PrimeKGwE_relation_embedding_path)
PrimeKG_entity_embed = np.load(PrimeKG_entity_embedding_path)
PrimeKG_relation_embed = np.load(PrimeKG_relation_embedding_path)

with open(relation_dict_path, "r") as f:
    relation_dict = dict(line.strip().split("\t")[::-1] for line in f)
    relation_dict = {k: int(v) for k, v in relation_dict.items()}

with open("pickle/PrimeKG/NER_ID_dict_cap_final.pickle", "rb") as f:
    name_to_idx = pickle.load(f)

with open("pickle/PrimeKG/biokdeid_type_map.pickle", "rb") as f:
    biokdeid_type_map = pickle.load(f)

with open("pickle/PrimeKG/DBRelations.no5.pickle", "rb") as f:
    DBRelations = pickle.load(f)

def load_entity_type_dict(type_path):
    entity_id2type = {}
    with open(type_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                eid = int(parts[0])
                etype = parts[2]
                entity_id2type[eid] = etype
    return entity_id2type

def compute_score(head, rel, tail, gamma=12.0):
    pi = 3.14159265358979323846
    phase_head, mod_head = torch.chunk(head, 2, dim=-1)
    phase_rel, mod_rel, bias_rel = torch.chunk(rel, 3, dim=-1)
    phase_tail, mod_tail = torch.chunk(tail, 2, dim=-1)
    embedding_range = (gamma + 2.0) / mod_head.shape[1]
    phase_head = phase_head / (embedding_range / pi)
    phase_rel = phase_rel / (embedding_range / pi)
    phase_tail = phase_tail / (embedding_range / pi)
    phase_score = (phase_head + phase_rel - phase_tail) / 2
    phase_score = torch.sum(torch.abs(torch.sin(phase_score)), dim=-1)
    mod_rel = torch.abs(mod_rel)
    bias_rel = torch.clamp(bias_rel, max=1)
    indicator = bias_rel < -mod_rel
    bias_rel[indicator] = -mod_rel[indicator]
    r_score = mod_head * (mod_rel + bias_rel) - mod_tail * (1 - bias_rel)
    r_score = torch.norm(r_score, dim=-1)
    score = gamma - (r_score + phase_score)
    return score

def load_id_mapping(entity_path):
    id2name = {}
    name2id = {}
    with open(entity_path) as f:
        for line in f:
            eid, name = line.strip().split("\t")
            id2name[int(eid)] = name
            name2id[name] = int(eid)
    return id2name, name2id

def get_topk_tail_entities(head_id, rel_id, entity_emb, relation_emb, entity_id2type=None, filter_type=None, topk=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity_emb = torch.tensor(entity_emb).to(device)
    relation_emb = torch.tensor(relation_emb).to(device)
    head = entity_emb[head_id].unsqueeze(0)
    rel = relation_emb[rel_id].unsqueeze(0)
    scores = compute_score(head.repeat(entity_emb.shape[0], 1),
                           rel.repeat(entity_emb.shape[0], 1),
                           entity_emb)
    if entity_id2type and filter_type:
        valid_indices = [i for i in range(len(entity_emb)) if entity_id2type.get(i) == filter_type]
        if not valid_indices:
            return []
        valid_scores = scores[valid_indices]
        topk_indices = valid_scores.topk(min(topk, len(valid_indices)), largest=True).indices
        topk_tail_ids = [valid_indices[i.item()] for i in topk_indices]
        topk_scores = [valid_scores[i].item() for i in topk_indices]
    else:
        topk_indices = scores.topk(topk, largest=True).indices
        topk_tail_ids = [i.item() for i in topk_indices]
        topk_scores = [scores[i].item() for i in topk_indices]
    return list(zip(topk_tail_ids, topk_scores))

def clean_pubtator_tags(snippets):
    """
    清理 PubTator 標記，同時保留實體文字。
    - 將 @CHEMICAL_xxx 等 placeholder 移除
    - 將 @@@<m>xxx</m>@@@ 換成 xxx
    - 過濾掉 None 或非字串輸入
    """
    cleaned = []
    for s in snippets:
        if not isinstance(s, str):
            continue  # 跳過 None 或非字串
        # 移除 @CHEMICAL_xxx, @DISEASE_xxx, @GENE_xxx 等標記
        s = re.sub(r"@(?:CHEMICAL|DISEASE|GENE|VARIANT|SPECIES|CELLLINE)_[^\s@]+", "", s)
        # 把 @@@<m>xxx</m>@@@ 換成 xxx
        s = re.sub(r"@@@<m>(.*?)</m>@@@", r"\1", s)
        s = re.sub(r"@@@", r"", s)
        s = re.sub(r"<m>(.*?)</m>", r"\1", s)
        # 去掉多餘空白
        s = re.sub(r"\s+", " ", s).strip()
        cleaned.append(s)
    return cleaned

def get_pubtator_snippets(disease, gene, topk=3, max_retry=5):
    query = f"{disease} and {gene}"
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text={requests.utils.quote(query)}"
    for attempt in range(max_retry):
        time.sleep(1)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"[!] 429 Too Many Requests. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            result_json = response.json()
            snippets = [item["text_hl"] for item in result_json.get("results", [])[:topk]]
            return snippets
        except Exception as e:
            print(f"[!] Failed to retrieve PubTator snippets for '{query}': {e}")
            if attempt == max_retry - 1:
                return []

def hake_score(h, r, t, gamma=12.0):
    pi = 3.14159265358979323846
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = torch.tensor(h).to(device)
    r = torch.tensor(r).to(device)
    t = torch.tensor(t).to(device)
    h_phase, h_mod = torch.chunk(h, 2, dim=-1)
    r_phase, r_mod, r_bias = torch.chunk(r, 3, dim=-1)
    t_phase, t_mod = torch.chunk(t, 2, dim=-1)
    embedding_range = (gamma + 2.0) / h_mod.shape[0]
    h_phase = h_phase / (embedding_range / pi)
    r_phase = r_phase / (embedding_range / pi)
    t_phase = t_phase / (embedding_range / pi)
    phase_score = torch.sum(torch.abs(torch.sin((h_phase + r_phase - t_phase) / 2)), dim=-1)
    r_mod_ = torch.abs(r_mod)
    r_bias_ = torch.clamp(r_bias, max=1)
    indicator = r_bias_ < -r_mod_
    r_bias_[indicator] = -r_mod_[indicator]
    r_score = h_mod * (r_mod_ + r_bias_) - t_mod * (1 - r_bias_)
    r_score = torch.norm(r_score, p=2)
    score = gamma - (r_score + phase_score)
    return score.item()

def soft_sigmoid(x, offset=-350, scale=100):
    return 1 / (1 + np.exp(-(x - offset) / scale))

def total_hake_score(G, path, entity_embed, relation_embed, relation_dict):
    scores = []
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        edge_id = G.get_eid(src, tgt, directed=True, error=False)
        if edge_id == -1:
            return -float('inf')
        h_id = int(G.vs[src]['node_id'])
        t_id = int(G.vs[tgt]['node_id'])
        r_type = G.es[edge_id]['relationship_type']
        r_idx = relation_dict.get(r_type)
        if r_idx is None:
            return -float('inf')
        raw_score = hake_score(entity_embed[h_id], relation_embed[r_idx], entity_embed[t_id])
        norm_score = soft_sigmoid(raw_score)
        scores.append(norm_score)
    if not scores:
        return 0
    product = np.prod(scores)
    geo_mean = product ** (1 / len(scores))
    return round(geo_mean, 4)

def normalize(text):
    return re.sub(r"\W+", "", str(text).strip().lower())

def find_exact_match(entity_name, target_type):
    q = normalize(entity_name)
    matched = set()
    for section in ["official name", "common name", "id"]:
        section_dict = name_to_idx.get(section, {})
        if q in section_dict:
            cands = section_dict[q]
            if isinstance(cands, str):
                cands = [cands]
            for cid in cands:
                if biokdeid_type_map.get(cid, "").lower() == target_type.lower():
                    matched.add(cid)
        if matched:
            break
    return list(matched)

def load_parameters(param_file):
    params = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            params[key.strip()] = value.strip()
    return params

def load_prompt_template(template_path, disease_name, payload):
    """
    Load the prompt template and replace placeholders:
      - [Disease] -> disease_name (str)
      - [JSON_Input] -> pretty-printed JSON derived from `payload`
    """
    import json
    with open(template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    json_text = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if isinstance(payload, (dict, list)) else str(payload)
    )

    prompt_filled = (
        prompt_template
        .replace("[Disease]", str(disease_name))
        .replace("[JSON_Input]", json_text)
    )
    return prompt_filled


def ask_gpt(messages, params):
    if params.get("OPENAI_API_KEY") and not params.get("AZURE_OPENAI_ENDPOINT"):
        client = openai.OpenAI(
            api_key=params["OPENAI_API_KEY"],
            base_url=params.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        deployment_name = params.get("OPENAI_MODEL") or params.get("MODEL") or "gpt-4o"
    else:
        openai.api_type = "azure"
        openai.api_base = params["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = params["API_KEY"]
        openai.api_version = params["API_VERSION"]
        deployment_name = params["DEPLOYMENT_NAME"]
        client = openai.AzureOpenAI(
            api_key=params["API_KEY"],
            api_version=params["API_VERSION"],
            azure_endpoint=params["AZURE_OPENAI_ENDPOINT"]
        )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0,
        max_completion_tokens=2048
    )
    return response.choices[0].message.content

def truncate_incomplete_json_array(text):
    pattern = r"\{[^{}]*\}"
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError("No complete objects found in input.")
    valid_objects = [m.group() for m in matches]
    fixed = "[\n  " + ",\n  ".join(valid_objects) + "\n]"
    return fixed

def load_name_map(name_dict_path):
    with open(name_dict_path, "rb") as f:
        name_dict = pickle.load(f)
    id_to_official = {}
    for norm_name, ids in name_dict.get("official name", {}).items():
        if not isinstance(ids, list):
            ids = [ids]
        for bid in ids:
            id_to_official[bid] = norm_name
    return id_to_official

def convert_path_ids_to_names(path_str, id_to_official):
    parts = re.split(r"(-\(.+?\)->)", path_str)
    new_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            name = id_to_official.get(part.strip(), part.strip())
            new_parts.append(name)
        else:
            new_parts.append(part)
    return "".join(new_parts)

def main(disease_name, disease_id, output_json, payload):
    """
    disease_name: string resolved from JSON (disease_match -> fallback disease)
    payload: the parsed JSON object (dict or list) to inject into [JSON_Input]
    """
    param_file = "parameter.gpt4o.txt"
    prompt_Candidate_Generation = "prompts/2.GPT4TargetGene.CandidateGeneration.txt"

    params = load_parameters(param_file)
    start_time = time.time()

    disease = disease_name.strip()
    DiseaseID = str(disease_id).strip()

    # --- NEW: load maps needed by downstream steps ---
    entity_id2type = load_entity_type_dict(entities_type_dict)
    id_to_official = load_name_map("pickle/PrimeKG/NER_ID_dict_cap_final.pickle")

    disease = disease_name.strip()
    DiseaseID = str(disease_id).strip()

    # Step 1. Candidate Generated by GPT
    print(disease + " - Step 1. Candidate Generated by GPT...")
    try:
        # Inject both [Disease] and [JSON_Input] using the payload
        reasoning_prompt = load_prompt_template(
            prompt_Candidate_Generation, disease, payload
        )
        reasoning_response = ask_gpt(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": reasoning_prompt},
            ],
            params,
        )
    except Exception as e:
        reasoning_response = f"[ERROR] {e}"

    # 修正 JSON（如有截斷）
    reasoning_response = truncate_incomplete_json_array(reasoning_response)
    try:
        json_reasoning_response = json.loads(reasoning_response)
    except Exception:
        # 若仍失敗就放空，避免中斷後續流程
        json_reasoning_response = []

    # 加入 ID 欄位（Gene）
    for item in json_reasoning_response:
        gene_name = item.get("Gene", "")
        matches = find_exact_match(gene_name, target_type="Gene")
        item["ID"] = matches[0] if matches else "[NOT_FOUND]"

    # Step 1.2. HAKE top-100 from PrimeKG and PrimeKGwE
    print(disease + " - Step 1.2. Candidate Generated by HAKE top-100 (PrimeKG + PrimeKGwE)...")

    existing_ids = {str(it.get("ID", "")) for it in json_reasoning_response if it.get("ID")}

    # 取得 PrimeKG 和 PrimeKGwE 的 top100
    topk_prime = get_topk_tail_entities(
        head_id=int(DiseaseID),
        rel_id=6,  # disease->gene relation
        entity_emb=PrimeKG_entity_embed,
        relation_emb=PrimeKG_relation_embed,
        entity_id2type=entity_id2type,
        filter_type="Gene",
        topk=100
    )
    topk_primewe = get_topk_tail_entities(
        head_id=int(DiseaseID),
        rel_id=6,
        entity_emb=PrimeKGwE_entity_embed,
        relation_emb=PrimeKGwE_relation_embed,
        entity_id2type=entity_id2type,
        filter_type="Gene",
        topk=100
    )

    # 合併兩個結果並去重
    merged_results = {}
    for rank, (gid, score) in enumerate(topk_prime, start=1):
        gid_str = str(gid)
        merged_results.setdefault(gid_str, {
            "Gene": id_to_official.get(gid_str, f"[{gid_str}]"),
            "Mechanism": "",
            "ID": gid_str,
            "_rank": []
        })
        merged_results[gid_str]["_rank"].append(f"PrimeKG_Rank:{rank}")

    for rank, (gid, score) in enumerate(topk_primewe, start=1):
        gid_str = str(gid)
        merged_results.setdefault(gid_str, {
            "Gene": id_to_official.get(gid_str, f"[{gid_str}]"),
            "Mechanism": "",
            "ID": gid_str,
            "_rank": []
        })
        merged_results[gid_str]["_rank"].append(f"PrimeKGwE_Rank:{rank}")

    # 將合併結果加入 json_reasoning_response，並避免重複
    for gid_str, record in merged_results.items():
        if gid_str in existing_ids:
            # 已存在 → 更新 Mechanism
            for item in json_reasoning_response:
                if str(item.get("ID", "")) == gid_str:
                    old_mech = item.get("Mechanism", "")
                    ranks = "; ".join(record["_rank"])
                    if ranks not in (old_mech or ""):
                        item["Mechanism"] = f"{old_mech}; {ranks}" if old_mech else ranks
        else:
            json_reasoning_response.append({
                "Gene": record["Gene"],
                "Mechanism": "; ".join(record["_rank"]),
                "ID": gid_str
            })
            existing_ids.add(gid_str)
            
    # Step 2.1. KG path（取前 10 依 HAKE 幾何平均分）
    print(disease + " - Step 2.1. KG path evidence...")
    nodeid_to_index = {str(v["node_id"]): v.index for v in DBRelations.vs}
    for item in json_reasoning_response:
        gene_id = str(item.get("ID", ""))
        dis_id = str(DiseaseID)
        if gene_id == "[NOT_FOUND]":
            print(f"[!] Skip KG evidence for {item.get('Gene', '')} because ID was not found.")
            item["KG_evidence"] = []
            continue
        if gene_id not in nodeid_to_index or dis_id not in nodeid_to_index:
            print(f"[!] Skip KG evidence for {gene_id} → {dis_id} (not found in graph)")
            item["KG_evidence"] = []
            continue

        source_idx = nodeid_to_index[gene_id]
        target_idx = nodeid_to_index[dis_id]
        paths = DBRelations.get_all_shortest_paths(source_idx, to=target_idx, mode="OUT")

        scored_paths = []
        for path in paths:
            parts = []
            for i in range(len(path) - 1):
                src = DBRelations.vs[path[i]]["name"]
                rel_id = DBRelations.get_eid(path[i], path[i + 1], directed=True, error=False)
                rel = DBRelations.es[rel_id]["relationship_type"] if rel_id != -1 else "?"
                parts.append(src)
                parts.append(f"-({rel})->")
            parts.append(DBRelations.vs[path[-1]]["name"])
            path_str = " ".join(parts)
            score = total_hake_score(DBRelations, path, PrimeKG_entity_embed, PrimeKG_relation_embed, relation_dict)
            scored_paths.append((score, path_str))
        scored_paths.sort(reverse=True, key=lambda x: x[0])
        top10_paths = [f"{p} [score={s:.4f}]" for s, p in scored_paths[:10]]

        # ID → 名稱 映射
        item["KG_evidence"] = [
            convert_path_ids_to_names(p.split(" [score=")[0], id_to_official) + f" [score={p.split('[score=')[-1]}"
            for p in top10_paths
        ]

    # Step 2.2. PubTator 片段
    print(disease + " - Step 2.2. PubTator Snippets...")
    for item in json_reasoning_response:
        gene = item.get("Gene", "")
        snippets = get_pubtator_snippets(disease, gene)
        item["snippet"] = clean_pubtator_tags(snippets)

    # Final：輸出（移除 Steps 3/4/5 相關欄位）
    with open(output_json, "w", encoding="utf-8") as f_out:
        for item in json_reasoning_response:
            item["Disease"] = disease
            # 不包含：SupportingEvidence / RisksOrLimitations / Confidence_score
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"[Time] Gene {disease} took {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KGxLM reasoning using a JSON file input that contains 'disease_match'."
    )
    parser.add_argument(
        "--json_input",
        required=True,
        help="Path to a JSON file (root object or first item) containing 'disease_match' (fallback to 'disease')."
    )
    parser.add_argument(
        "--output",
        help="Output file (default: output.jsonl)",
        default="output.jsonl"
    )
    args = parser.parse_args()

    # Avoid overwriting output
    if Path(args.output).exists():
        print(f"⚠️ Output file already exists: {args.output}. Skipping and exiting.")
        exit(0)

    # Read JSON and resolve disease name
    try:
        with open(args.json_input, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object or a non-empty array of objects.")
        disease_name = (payload.get("disease_match") or payload.get("disease") or "").strip()
        if not disease_name:
            raise ValueError("No 'disease_match' (or fallback 'disease') found in JSON.")
        print(f"[INFO] Using disease from JSON {args.json_input}: {disease_name}")
    except Exception as e:
        print(f"❌ Failed to read disease from JSON: {e}")
        exit(1)

    # Resolve Disease ID
    disease_ids = find_exact_match(disease_name, target_type="Disease")
    if not disease_ids:
        print(f"❌ Cannot find ID for disease: {disease_name}")
        exit(1)
    disease_id = disease_ids[0]

    # Run
    main(disease_name, disease_id, args.output, payload)

    
"""
time python 2.KGxLM.disease2gene.candidate.py \
  --json_input input/mCRC.json \
  --output output/disease2gene/mcrc.disease2gene.candidate.jsonl
  
time python 2.KGxLM.disease2gene.candidate.py \
  --json_input input/melanoma.json \
  --output output/disease2gene/melanoma.disease2gene.candidate.jsonl
"""
  