import pandas as pd
import openai
import json
import time
import tiktoken
import pickle
import re
import numpy as np
import torch
import igraph as ig
import requests
from pathlib import Path
from tqdm import tqdm
import argparse    
from typing import List, Dict
import csv
    
# 路徑設定
PrimeKGwE_entity_embedding_path = Path("HAKE_model/PrimeKGwE/entity_embedding.npy")
PrimeKGwE_relation_embedding_path = Path("HAKE_model/PrimeKGwE/relation_embedding.npy")
PrimeKG_entity_embedding_path = Path("HAKE_model/PrimeKG/entity_embedding.npy")
PrimeKG_relation_embedding_path = Path("HAKE_model/PrimeKG/relation_embedding.npy")
relation_dict_path = Path("data/PrimeKG/relations.dict")
entity_dict="data/PrimeKG/entities.dict"
entities_type_dict="data/PrimeKG/entities.dict.type"
    

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

def _norm_drug_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()
    
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
    device = head.device

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

    # 加入類型過濾
    if entity_id2type and filter_type:
        valid_indices = [i for i in range(len(entity_emb)) if entity_id2type.get(i) == filter_type]
        valid_scores = scores[valid_indices]
        topk_indices = valid_scores.topk(min(topk, len(valid_indices)), largest=True).indices
        topk_tail_ids = [valid_indices[i.item()] for i in topk_indices]
        topk_scores = [valid_scores[i].item() for i in topk_indices]
    else:
        topk_indices = scores.topk(topk, largest=True).indices
        topk_tail_ids = [i.item() for i in topk_indices]
        topk_scores = [scores[i].item() for i in topk_indices]

    return list(zip(topk_tail_ids, topk_scores))

def run_topk(input_tsv, output_tsv, entity_emb_file, relation_emb_file, entity_dict, relation_dict, rel_id, id2type, tail_type_filter):
    df = pd.read_csv(input_tsv, sep="\t")
    entity_emb = np.load(entity_emb_file)
    relation_emb = np.load(relation_emb_file)
    id2name, name2id = load_id_mapping(entity_dict)

    output = []
    for _, row in df.iterrows():
        disease = str(row["Disease"])
        disease_ids = str(row["DiseaseID"]).split("|")
        disease_ids = list(set([int(x) for x in disease_ids]))  # 去重

        for dis_id in disease_ids:
            topk_results = get_topk_tail_entities(
                dis_id, rel_id, entity_emb, relation_emb,
                entity_id2type=id2type, filter_type=tail_type_filter, topk=100
            )

            rank=1
            for tail_id, score in topk_results:
                output.append({
                    "Disease": disease,
                    "DiseaseID": dis_id,
                    "DrugID": tail_id,
                    "Drug": id2name.get(tail_id, f"ID:{tail_id}"),
                    "Score": round(score, 4),
                    "Rank": rank
                })
                #print(output[-1])
                rank=rank+1

    pd.DataFrame(output).to_csv(output_tsv, sep="\t", index=False)
    print(f"✅ Top-100 results written to: {output_tsv}")
    
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

def get_pubtator_snippets(disease, drug, topk=3, max_retry=5):
    query = f"{disease} and {drug}"
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
    else:
        product = np.prod(scores)
        geo_mean = product ** (1 / len(scores))
        return round(geo_mean, 4)
        
def normalize(text):
    return re.sub(r"\W+", "", str(text).strip().lower())
    
# 回傳符合 type 的 accession（精確 match）
def find_exact_match(entity_name, target_type):
    
    q = normalize(entity_name)  # 標準化查詢名稱
    matched = set()

    for section in ["official name", "common name", "id"]:
        section_dict = name_to_idx.get(section, {})
        if q in section_dict:
            cands = section_dict[q]
            if isinstance(cands, str):  # 若為單一字串轉成 list
                cands = [cands]
            for cid in cands:
                # 比對實體類型
                if biokdeid_type_map.get(cid, "").lower() == target_type.lower():
                    matched.add(cid)
        if matched:
            break  # 找到就結束（先匹配的區段優先）
    return list(matched)
    
# 讀取參數檔
def load_parameters(param_file):
    params = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split("\t")
                params[key] = value
    return params

# 讀取提示模板
def load_prompt_template(template_path, disease_name, payload):
    """
    Load the prompt template and replace placeholders:
      - [Disease] -> disease_name (str)
      - [JSON_Input] -> pretty-printed JSON derived from `payload`
    """
    import json

    with open(template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Prepare JSON text for insertion
    if isinstance(payload, (dict, list)):
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        # If payload is already a JSON string or any other type, coerce to str
        json_text = str(payload)

    # Replace placeholders
    prompt_filled = (
        prompt_template
        .replace("[Disease]", str(disease_name))
        .replace("[JSON_Input]", json_text)
    )

    return prompt_filled
    
# 呼叫 GPT 模型
def ask_gpt(messages, params):
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

def extract_confidence_score(text):
    match = re.search(r"confidence_score\s*:\s*([0-1](?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Confidence score not found.")
        
def truncate_incomplete_json_array(text):
    """
    將不完整的 JSON array 修剪為有效格式，只保留完整的 {...} 區段，最後自動補上 ]。
    """
    # 尋找所有完整的 {...} 區段
    pattern = r"\{[^{}]*\}"
    matches = list(re.finditer(pattern, text))

    if not matches:
        raise ValueError("No complete objects found in input.")

    # 組合成新的陣列
    valid_objects = [m.group() for m in matches]
    fixed = "[\n  " + ",\n  ".join(valid_objects) + "\n]"
    return fixed
        
def load_name_map(name_dict_path):
    with open(name_dict_path, "rb") as f:
        name_dict = pickle.load(f)
    
    # 建立 biokde_id → official_name 映射
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
            name = id_to_official.get(part.strip(), part.strip())  # fallback 用原 ID
            new_parts.append(name)
        else:
            new_parts.append(part)
    return "".join(new_parts)
    
# ---------- FDA labels helpers ----------
def load_fda_labels_tsv(path: str) -> List[Dict[str, str]]:
    """
    Expect columns:
    brand_name  generic_name  indications_and_usage  contraindications  warnings
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # normalize headers for safety
        field_map = {k: k.strip() for k in reader.fieldnames or []}
        for row in reader:
            rows.append({
                "brand_name": (row.get("brand_name") or "").strip(),
                "generic_name": (row.get("generic_name") or "").strip(),
                "indications_and_usage": (row.get("indications_and_usage") or "").strip(),
                "contraindications": (row.get("contraindications") or "").strip(),
                "warnings": (row.get("warnings") or "").strip(),
            })
    return rows

def _merge_topk(json_reasoning_response, topk_list, tag_prefix,
                id_to_official, existing_ids, existing_names, norm_func):
    """
    將 topk_list (drug_idx, score) 合併進 json_reasoning_response，
    使用 id/name 去重，並更新 Mechanism 標籤。
    回傳更新後的 json_reasoning_response, existing_ids, existing_names
    """
    for rank, (drug_idx, score) in enumerate(topk_list, start=1):
        drug_id = str(drug_idx).strip()
        rank_str = f"{tag_prefix}_Rank:{rank}"
        official_name = id_to_official.get(drug_id, f"[{drug_id}]")
        name_norm = norm_func(official_name)

        if drug_id in existing_ids:
            for item in json_reasoning_response:
                if str(item.get("ID", "")).strip() == drug_id:
                    old_mech = item.get("Mechanism", "")
                    if rank_str not in (old_mech or ""):
                        item["Mechanism"] = f"{old_mech}; {rank_str}" if old_mech else rank_str
            continue

        if name_norm in existing_names:
            for item in json_reasoning_response:
                if norm_func(item.get("Drug", "")) == name_norm:
                    old_mech = item.get("Mechanism", "")
                    if rank_str not in (old_mech or ""):
                        item["Mechanism"] = f"{old_mech}; {rank_str}" if old_mech else rank_str
                    if not item.get("ID") or item["ID"] == "[NOT_FOUND]":
                        item["ID"] = drug_id
                        existing_ids.add(drug_id)
            continue

        json_reasoning_response.append({
            "Drug": official_name,
            "Mechanism": rank_str,
            "ID": drug_id
        })
        existing_ids.add(drug_id)
        existing_names.add(name_norm)

    return json_reasoning_response, existing_ids, existing_names

def _strong_norm_name(s: str) -> str:
        # stronger normalization: lowercase, collapse whitespace, strip non-alphanumerics
        s = " ".join((s or "").split()).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", s)

# 主程式
def main(disease_name, disease_id, payload, output_json):
    # helpers
    def _norm_name(s: str) -> str:
        return " ".join((s or "").split()).strip().lower()

    id2type = load_entity_type_dict(entities_type_dict)
    id_to_official = load_name_map("pickle/PrimeKG/NER_ID_dict_cap_final.pickle")

    # files and params
    param_file = "parameter.gpt4o.txt"
    params = load_parameters(param_file)

    start_time = time.time()

    disease = disease_name.strip()
    DiseaseID = str(disease_id).strip()

    # Step 1. Candidate Drugs from File (replaces 1.1 LLM + 1.2 HAKE)
    print(disease + " - Step 1. Candidate Drugs from file...")

    if not args.drug_list_file:
        raise RuntimeError("Please provide --drug_list_file pointing to a text file with one drug name per line.")

    json_reasoning_response = build_candidates_from_drug_file(
        args.drug_list_file,
        name_to_idx=name_to_idx,
        id_to_official=id_to_official
    )

    print(disease + f" - Loaded {len(json_reasoning_response)} unique candidates from file.")
    # Step 2. Retrieval of Literature-Based Evidence
    # Step 2.1+2.2. KG paths scored by both KGs; store separately and keep backward-compatible KG_evidence
    print(disease + " - Step 2. KG path evidence (PrimeKG, PrimeKGwE)...")

    nodeid_to_index = {str(v["node_id"]): v.index for v in DBRelations.vs}

    def path_to_id_string(G, path_idx_list):
        parts = []
        for i in range(len(path_idx_list) - 1):
            src_id = str(G.vs[path_idx_list[i]]["node_id"])
            rel_eid = G.get_eid(path_idx_list[i], path_idx_list[i + 1], directed=True, error=False)
            rel = G.es[rel_eid]["relationship_type"] if rel_eid != -1 else "?"
            parts.append(src_id)
            parts.append(f"-({rel})->")
        parts.append(str(G.vs[path_idx_list[-1]]["node_id"]))
        return " ".join(parts)

    for item in json_reasoning_response:
        drug_id = str(item.get("ID", "")).strip()
        dis_id = str(DiseaseID)

        # default empty lists
        item["KG_evidence"] = []

        if drug_id == "[NOT_FOUND]":
            print(f"[!] Skip KG evidence for {item.get('Drug', '')} because ID was not found.")
            continue

        if drug_id not in nodeid_to_index or dis_id not in nodeid_to_index:
            print(f"[!] Skip KG evidence for {drug_id} -> {dis_id} (not found in graph)")
            continue

        source_idx = nodeid_to_index[drug_id]
        target_idx = nodeid_to_index[dis_id]
        paths = DBRelations.get_all_shortest_paths(source_idx, to=target_idx, mode="OUT")

        scored_prime = []
        for p in paths:
            pid_str = path_to_id_string(DBRelations, p)
            s1 = total_hake_score(DBRelations, p, PrimeKG_entity_embed, PrimeKG_relation_embed, relation_dict)
            scored_prime.append((s1, pid_str))
           
        scored_prime.sort(key=lambda x: x[0], reverse=True)
        
        top10_prime = []
        for s, pid in scored_prime[:10]:
            named = convert_path_ids_to_names(pid, id_to_official)
            top10_prime.append(f"{named} [score={s:.4f}]")

        item["KG_evidence"] = top10_prime  # keep old field for compatibility

    # Step 2.3. PubTator Snippets
    print(disease + " - Step 2.3. PubTator Snippets...")
    for item in json_reasoning_response:
        drug = item.get("Drug", "")
        snippets = get_pubtator_snippets(disease, drug)
        item["snippet"] = clean_pubtator_tags(snippets)

    # Final step. output result
    with open(output_json, "w", encoding="utf-8") as f_out:
        for item in json_reasoning_response:
            item["Disease"] = disease
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[Time] Drug {disease} took {elapsed:.2f} seconds.")

# NEW: read drug list and build candidate items from file
# 由於 find_exact_match 需要目標類型，我們在此處假設藥物為 "Chemical"。
TARGET_DRUG_TYPE = "Chemical" 

def build_candidates_from_drug_file(drug_file_path, name_to_idx, id_to_official):
    """
    Read a text file with one drug name per line (empty lines/# comments ignored).
    Look up PrimeKG entity ID using find_exact_match (target type: Chemical); 
    otherwise mark as [NOT_FOUND].
    Returns a list of dict items compatible with downstream pipeline.
    """
    candidates = []
    try:
        with open(drug_file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
    except Exception as e:
        # 使用 runtimeerror 確保錯誤處理與原程式碼一致
        raise RuntimeError(f"Failed to read drug list file: {drug_file_path} ({e})")

    for raw in lines:
        if not raw or raw.startswith("#"):
            continue
        nm_raw = raw
        
        # 🎯 新增: 使用 find_exact_match 查找 ID
        # 它處理了 name_to_idx 的多層結構，並過濾了實體類型。
        matched_ids = find_exact_match(nm_raw, target_type=TARGET_DRUG_TYPE)
        
        ent_id = matched_ids[0] if matched_ids else None  # 使用找到的第一個 ID
        
        if ent_id is not None:
            id_str = str(ent_id)
            # 使用 official name
            # 必須轉換為 int，因為 id_to_official 期望 int 鍵
            official = id_to_official.get(int(ent_id), nm_raw) 
        else:
            id_str = "[NOT_FOUND]"
            official = nm_raw
            # 這裡可以選擇性地添加一個列印語句來追蹤未找到的藥物
            # print(f"[WARN] Drug '{nm_raw}' not found in PrimeKG (type: {TARGET_DRUG_TYPE})")

        # 建立最小化的候選項目
        item = {
            "Drug": official,
            "ID": id_str,
            "Mechanism": "",
            "Mechanism_of_Action": "",
            "Source": "DrugListFile"
        }
        candidates.append(item)

    # Deduplicate & aggregate logic (保持與原程式碼一致)
    merged = {}
    import re  
    for it in candidates:
        id_str = (it.get("ID") or "").strip()
        # 注意: 這裡使用 _strong_norm_name，請確保它在腳本中是可用的 helper 函數
        nm_norm2 = _strong_norm_name(it.get("Drug", "")) 

        key = ("ID", id_str) if id_str and id_str != "[NOT_FOUND]" else ("NAME", nm_norm2)

        if key not in merged:
            merged[key] = dict(it)  
        else:
            base = merged[key]

            # Prefer to upgrade ID if previously NOT_FOUND
            if (base.get("ID") in ("", "[NOT_FOUND]")) and (id_str not in ("", "[NOT_FOUND]")):
                base["ID"] = id_str

            # Merge Mechanism (unique, semicolon-joined)
            m1 = base.get("Mechanism", "") or ""
            m2 = it.get("Mechanism", "") or ""
            mech_set = set()
            if m1:
                mech_set.update([x.strip() for x in re.split(r"\s*;\s*", m1) if x.strip()])
            if m2:
                mech_set.update([x.strip() for x in re.split(r"\s*;\s*", m2) if x.strip()])
            base["Mechanism"] = "; ".join(sorted(mech_set)) if mech_set else ""

            # Merge Mechanism_of_Action (unique, pipe-joined) across duplicates
            moa1 = base.get("Mechanism_of_Action", "") or ""
            moa2 = it.get("Mechanism_of_Action", "") or ""
            moa_set = set()
            if moa1:
                moa_set.update([p.strip() for p in moa1.split("|") if p.strip()])
            if moa2:
                moa_set.update([p.strip() for p in moa2.split("|") if p.strip()])

            # Add Mechanism tokens into MOA (unique)
            if mech_set:
                moa_set.update(mech_set)

            # Materialize MOA
            base["Mechanism_of_Action"] = " | ".join(sorted(moa_set)) if moa_set else ""
            # print(base["Mechanism_of_Action"]) # 註釋掉原有的 print

    return list(merged.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KGxLM reasoning using a JSON file input that contains 'disease_match'."
    )
    # Simplified: only accept an input JSON file (no disease name arg).
    parser.add_argument(
        "--json_input",
        help="Path to JSON file (same as positional)."
    )

    # NEW: required drug list file
    parser.add_argument(
        "--drug_list_file",
        required=True,
        help="Path to a text file with one drug name per line; lines starting with # are ignored."
    )
    parser.add_argument(
        "--output",
        help="Output file (default: output.jsonl)",
        default="output.jsonl"
    )
    parser.add_argument(
        "--fda_labels_tsv",
        default="DB/FDA-approved/labels.tsv",
        help="Path to FDA labels.tsv (brand_name, generic_name, indications_and_usage, contraindications, warnings)"
    )
    args = parser.parse_args()

    # Check if output file already exists
    if Path(args.output).exists():
        print(f"⚠️ Output file already exists: {args.output}. Skipping and exiting.")
        exit(0)

    # 2.5) FDA labels enrichment
    fda_rows = []
    try:
        fda_rows = load_fda_labels_tsv(args.fda_labels_tsv)
    except FileNotFoundError:
        print(f"[!] FDA labels file not found: {args.fda_labels_tsv} (continuing without FDA fields)")
    except Exception as e:
        print(f"[!] Failed to load FDA labels: {e} (continuing without FDA fields)")

    # Read disease name from JSON (prefer 'disease_match', fallback to 'disease')
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

    disease_id = disease_ids[0]  # Use the first ID; adjust if you need multi-ID handling
    main(disease_name, disease_id, payload, args.output)

    
"""
time python 1.KGxLM.disease2drug.candidate.list.py \
  --json_input input/melanoma_txGNN.json \
  --drug_list_file input/melanoma_txGNN.drug.txt \
  --output output/disease2drug/melanoma_txGNN.disease2drug.candidate.jsonl
"""