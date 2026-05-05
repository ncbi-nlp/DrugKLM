#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


PV_KEYS = [
    "adjusted_p_value", "adj_p_value", "padj", "fdr", "qvalue", "q_value",
    "adjP", "AdjP", "adj.P.Val", "FDR", "qvalueStorey", "qval", "p.adjust"
]

TERM_KEYS = [
    "term", "Term", "pathway", "Pathway", "Description", "name", "Name",
    "gs_name", "setName", "ID", "term_name"
]

DB_KEYS = [
    "Database", "database", "db", "source", "Source"
]

GENE_KEYS = [
    "OverlapGenes", "overlap_genes", "overlap.genes", "overlapping_genes",
    "Genes", "genes", "LEADING_EDGE", "leadingEdge"
]


def read_params_tsv(path: str) -> Dict[str, str]:
    kv = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                kv[parts[0].strip()] = "\t".join(parts[1:]).strip()
    # Accept either OpenAI- or Azure-mode credentials.
    if kv.get("OPENAI_API_KEY") and not kv.get("AZURE_OPENAI_ENDPOINT"):
        return kv
    required = ["API_KEY", "API_VERSION", "AZURE_OPENAI_ENDPOINT", "DEPLOYMENT_NAME"]
    missing = [k for k in required if not kv.get(k)]
    if missing:
        raise ValueError(f"Missing Azure keys in param file: {missing}")
    return kv


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "NA"


def detect_column(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None


def normalize_gene_list(x: str, cap: int = 50) -> str:
    toks = re.split(r"[^A-Za-z0-9\-]+", str(x))
    toks = [t for t in toks if t]
    out = []
    seen = set()
    for t in toks:
        u = t.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= cap:
            break
    return ", ".join(out)


def prefilter_gsea(df: pd.DataFrame,
                   alpha: float,
                   top_k: int,
                   db_whitelist: Optional[List[str]] = None
                   ) -> Tuple[pd.DataFrame, str, str, Optional[str], Dict[str, str]]:
    # 1) 找調整後 p 值欄位
    pval_col = detect_column(df, PV_KEYS)
    if not pval_col:
        # fallback：找 raw p-value
        raw_p_candidates = ["p-value", "pvalue", "p_value", "PValue", "P.Value"]
        pval_col = detect_column(df, raw_p_candidates)

    term_col = detect_column(df, TERM_KEYS)
    db_col = detect_column(df, DB_KEYS)

    if not pval_col or not term_col:
        bad_cols = {"pval_col": pval_col, "term_col": term_col}
        raise ValueError(f"Could not find required GSEA columns: {bad_cols}")

    df2 = df.copy()
    if db_whitelist and db_col:
        df2 = df2[df2[db_col].astype(str).isin(db_whitelist)].copy()

    # 2) 過濾數值
    df2 = df2[pd.to_numeric(df2[pval_col], errors="coerce").notnull()]
    df2[pval_col] = df2[pval_col].astype(float)
    df2 = df2[df2[pval_col] < alpha].copy()
    if df2.empty:
        return df2, pval_col, term_col, db_col, {}

    # 3) 排序 + top_k
    df2 = df2.sort_values(pval_col, ascending=True).head(top_k).copy()

    # 4) 取出基因清單欄位
    gcol = detect_column(df2, GENE_KEYS + ["overlapping genes"])
    if gcol:
        df2["OverlapGenes"] = df2[gcol].astype(str).map(normalize_gene_list)
    else:
        df2["OverlapGenes"] = ""

    path2genes = {str(r[term_col]): str(r["OverlapGenes"]) for _, r in df2.iterrows()}
    return df2, pval_col, term_col, db_col, path2genes


def make_prompt_snippet(df_sig: pd.DataFrame,
                        term_col: str,
                        pval_col: str,
                        db_col: Optional[str]) -> str:
    disp_cols = [term_col, pval_col]
    if db_col:
        disp_cols.append(db_col)
    disp_cols.append("OverlapGenes")

    df_disp = df_sig[disp_cols].rename(columns={
        term_col: "Term",
        pval_col: "AdjP",
        (db_col if db_col else "Database"): "Database"
    })
    # if db_col is None, the rename above will introduce a "Database" mapping from a missing key; fix
    if "Database" not in df_disp.columns and db_col is None:
        pass  # nothing to do
    return df_disp.to_csv(sep="\t", index=False)


def fill_template_square_brackets(template: str,
                                  disease: str,
                                  drug: str,
                                  direction: str,
                                  alpha: float,
                                  topk: int,
                                  gsea_snippet: str) -> str:
    text = template
    repl = {
        "[DISEASE]": disease,
        "[DRUG]": drug,
        "[DIRECTION]": direction,
        "[ALPHA]": str(alpha),
        "[TOPK]": str(topk),
        "[gsea_results]": gsea_snippet
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def azure_chat_complete(
    params: Dict[str, str],
    prompt_text: str,
    retries: int = 5,
    delay_sec: float = 2.0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    jitter_min: int = 1,
    jitter_max: int = 10,
) -> str:
    """
    Robust Azure OpenAI Chat Completions with retry.
    - Retries on connection errors/timeouts and HTTP 429 (rate limit) or >=500 (server errors).
    - Sleeps a RANDOM 1–10 seconds between attempts (configurable by jitter_min/max),
      and respects Retry-After header when present.
    - Sleep duration each retry = max(delay_sec, retry_after, random(jitter_min..jitter_max)).

    Args keep backward compatibility with original signature.
    """
    # Auto-detect backend: OpenAI if OPENAI_API_KEY set, else Azure.
    if params.get("OPENAI_API_KEY") and not params.get("AZURE_OPENAI_ENDPOINT"):
        base_url = params.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = params.get("OPENAI_MODEL") or params.get("MODEL") or "gpt-4o"
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {params['OPENAI_API_KEY']}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that outputs STRICT JSON only."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "n": 1,
        }
    else:
        url = (
            params["AZURE_OPENAI_ENDPOINT"].rstrip("/")
            + f"/openai/deployments/{params['DEPLOYMENT_NAME']}/chat/completions"
            + f"?api-version={params['API_VERSION']}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": params["API_KEY"],
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that outputs STRICT JSON only."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "n": 1,
        }

    def _parse_retry_after(resp) -> float:
        # Retry-After can be seconds or HTTP-date; we only handle seconds here.
        try:
            ra = resp.headers.get("Retry-After")
            if not ra:
                return 0.0
            return float(re.sub(r"[^\d\.]", "", str(ra))) if re.search(r"\d", ra) else 0.0
        except Exception:
            return 0.0

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

            # Success path
            if 200 <= resp.status_code < 300:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content
                raise RuntimeError("Empty response content")

            # Transient HTTP errors eligible for retry
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = RuntimeError(f"Azure error {resp.status_code}: {resp.text[:500]}")
                if attempt < retries:
                    retry_after = _parse_retry_after(resp)
                    sleep_sec = max(delay_sec, retry_after, float(random.randint(jitter_min, jitter_max)))
                    sys.stderr.write(
                        f"[WARN] HTTP {resp.status_code} (attempt {attempt}/{retries}). "
                        f"Sleeping {sleep_sec:.1f}s before retry...\n"
                    )
                    time.sleep(sleep_sec)
                    continue
                raise last_err

            # Other 4xx: do not retry
            raise RuntimeError(f"Azure error {resp.status_code}: {resp.text[:500]}")

        except requests.exceptions.RequestException as e:
            # Network / connection / timeout -> retry
            last_err = e
            if attempt < retries:
                sleep_sec = max(delay_sec, float(random.randint(jitter_min, jitter_max)))
                sys.stderr.write(
                    f"[WARN] Connection error (attempt {attempt}/{retries}): {e}. "
                    f"Sleeping {sleep_sec:.1f}s before retry...\n"
                )
                time.sleep(sleep_sec)
                continue
            raise

        except Exception as e:
            # Unknown error: no retry (keep consistent with your original behavior)
            raise

    # Should not reach here
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")



def extract_first_json(text: str) -> Optional[str]:
    # try to find first balanced JSON object or array
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            top = stack[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                stack.pop()
                if not stack and start is not None:
                    return text[start:i+1]
    # fallback: whole text if it looks like JSON
    t = text.strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t
    return None


def parse_model_json(text: str) -> dict:
    cand = extract_first_json(text)
    if not cand:
        raise ValueError("No JSON object/array found in model output")
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        # very light repair: remove trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", cand)
        return json.loads(fixed)


def clean_genes_field(s: str, cap: int = 20) -> str:
    toks = [t for t in re.split(r"[,;/\s]+", str(s)) if t]
    out = []
    seen = set()
    for t in toks:
        u = re.sub(r"[^A-Za-z0-9\-]", "", t.upper())
        if u and u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= cap:
            break
    return ", ".join(out)


def normalize_output(obj: dict,
                     drug: str,
                     disease: str,
                     direction: str,
                     path2genes: Dict[str, str]) -> dict:
    sp = obj.get("selected_pathways", [])

    if isinstance(sp, dict):
        sp = [{"Pathway": k, "PathwayID": k, "Evidence": v, "Genes": ""} for k, v in sp.items()]
    elif isinstance(sp, list):
        norm = []
        for it in sp:
            
            #{
            #    'Pathway': 'actin filament network formation', 
            #    'PathwayID': 'GO:0051639', 
            #    'Evidence': 'This pathway is crucial for cell motility and structure, which can be affected in cancer.', 
            #    'Genes': 'CARMIL1, COBL, LCP1'
            #}
            
            p = (it.get("Pathway") or it.get("pathway") or "").strip()
            p_id = (it.get("PathwayID") or it.get("PathwayID") or "").strip()
            e = (it.get("Evidence") or it.get("evidence") or "").strip()
            g = (it.get("Genes") or it.get("genes") or "").strip()
            
            # --- 新增：解析 Pathway + GO ID ---
            norm.append({"Pathway": p, "PathwayID": p_id, "Evidence": e, "Genes": g})
        sp = norm
    else:
        sp = []

    for it in sp:
        if not it.get("Genes"):
            it["Genes"] = path2genes.get(it.get("Pathway", ""), "")
        it["Genes"] = clean_genes_field(it.get("Genes", "")) or path2genes.get(it.get("Pathway", ""), "")

    sp = [it for it in sp if it.get("Pathway")]

    obj["selected_pathways"] = sp
    obj["drug"] = obj.get("drug") or drug
    obj["disease"] = obj.get("disease") or disease
    obj["direction"] = obj.get("direction") or direction
    return obj


def main():
    ap = argparse.ArgumentParser(description="Summarize GSEA with LLM and output JSONL")
    ap.add_argument("--disease", required=True)
    ap.add_argument("--rank_tsv", required=True, help="TSV with columns Drug, Direction")
    ap.add_argument("--gsea_dir", required=True, help="Directory containing <Drug>.<Direction>.GSEA.tsv files")
    ap.add_argument("--prompt", required=True, help="Prompt template file (uses [PLACEHOLDERS])")
    ap.add_argument("--param_file", required=True, help="TSV with Azure keys")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--limit_pathways", type=int, default=0, help="Unused; kept for compatibility")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--delay_sec", type=float, default=2.0)
    ap.add_argument("--db_whitelist", type=str, default="", help="Comma-separated whitelist of DB names")
    args = ap.parse_args()

    disease = args.disease
    rank_tsv = args.rank_tsv
    gsea_dir = args.gsea_dir
    prompt_path = args.prompt
    params_path = args.param_file
    out_path = args.output
    alpha = args.alpha
    top_k = args.top_k
    retries = args.retries
    delay_sec = args.delay_sec
    db_whitelist = [s.strip() for s in args.db_whitelist.split(",") if s.strip()] if args.db_whitelist else None

    params = read_params_tsv(params_path)

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")

    df_rank = pd.read_csv(rank_tsv, sep="\t", dtype=str).fillna("")

    if "Drug" not in df_rank.columns:
        raise ValueError("rank_tsv must contain column: Drug")

    for idx, row in df_rank.iterrows():
        drug = row["Drug"].strip()
        if not drug:
            continue

        # 對每個 drug 檢查 up / down 基因
        for direction, colname in [("up", "TopUpGenes"), ("down", "TopDownGenes")]:
            if colname not in df_rank.columns:
                # 兼容舊版有 Direction 欄位的情況
                if "Direction" in df_rank.columns:
                    if row["Direction"].strip().lower() != direction:
                        continue
                else:
                    continue  # 沒有對應欄位就跳過
            else:
                # 如果沒有基因，直接跳過
                if not row[colname] or row[colname].strip() == "":
                    continue

            sdrug = sanitize_filename(drug)
            gsea_path = os.path.join(gsea_dir, f"{sdrug}.{direction}.GSEA.tsv")
            if not os.path.exists(gsea_path):
                # try unsanitized as fallback
                alt_path = os.path.join(gsea_dir, f"{drug}.{direction}.GSEA.tsv")
                if os.path.exists(alt_path):
                    gsea_path = alt_path
                else:
                    sys.stderr.write(f"[WARN] GSEA file not found: {gsea_path}\n")
                    continue

            try:
                df_gsea = pd.read_csv(gsea_path, sep="\t", dtype=str).fillna("")
                df_sig, pval_col, term_col, db_col, path2genes = prefilter_gsea(
                    df_gsea, alpha=alpha, top_k=top_k, db_whitelist=db_whitelist
                )
                if df_sig.empty:
                    sys.stderr.write(f"[INFO] No significant rows for {drug} {direction}\n")
                    continue

                snippet = make_prompt_snippet(df_sig, term_col, pval_col, db_col)
                prompt_text = fill_template_square_brackets(
                    template=template,
                    disease=disease,
                    drug=drug,
                    direction=direction,
                    alpha=alpha,
                    topk=top_k,
                    gsea_snippet=snippet,
                )

                raw_text = azure_chat_complete(
                    params=params,
                    prompt_text=prompt_text,
                    retries=retries,
                    delay_sec=delay_sec,
                    temperature=0.0,
                    max_tokens=2048,
                )

                obj = parse_model_json(raw_text)
                obj = normalize_output(obj, drug, disease, direction, path2genes)

                fout.write(json.dumps(obj, ensure_ascii=True) + "\n")
                fout.flush()
                sys.stderr.write(f"[OK] {drug} {direction}\n")

            except Exception as e:
                fallback = {
                    "drug": drug,
                    "disease": disease,
                    "direction": direction,
                    "selected_pathways": [],
                    "raw_output": str(e),
                }
                fout.write(json.dumps(fallback, ensure_ascii=True) + "\n")
                fout.flush()
                sys.stderr.write(f"[ERR] {drug} {direction}: {e}\n")

    fout.close()



if __name__ == "__main__":
    main()

"""
python 4.4.LINCS.GPT.py \
  --disease glioblastoma \
  --rank_tsv output/LINCS/LINCS.search.glioblastoma.rank.tsv \
  --gsea_dir output/LINCS_enrich/glioblastoma \
  --prompt prompts/4.4.GSEA_Summarizer.txt \
  --param_file parameter.gpt4o.txt \
  --output output/LINCS_enrich_summary/glioblastoma.GSEA.summary.jsonl

python 4.4.LINCS.GPT.py \
  --disease melanoma \
  --rank_tsv output/LINCS/melanoma/LINCS.search.melanoma.rank.tsv \
  --gsea_dir output/LINCS_enrich/melanoma \
  --prompt prompts/4.4.GSEA_Summarizer.txt \
  --param_file parameter.gpt4o.txt \
  --output output/LINCS_enrich_summary/melanoma.GSEA.summary.jsonl
  
python 4.4.LINCS.GPT.py \
  --disease mcrc \
  --rank_tsv output/LINCS/mcrc/LINCS.search.mcrc.rank.tsv \
  --gsea_dir output/LINCS_enrich/mcrc \
  --prompt prompts/4.4.GSEA_Summarizer.txt \
  --param_file parameter.gpt4o.txt \
  --output output/LINCS_enrich_summary/mcrc.GSEA.summary.jsonl
  
"""