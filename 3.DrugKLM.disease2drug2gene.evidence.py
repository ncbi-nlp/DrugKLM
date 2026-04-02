#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage GPT pipeline:
1) Use GPT to select top-K most relevant genes given per-gene integrated evidence
2) Use GPT once to summarize evidence for those K genes in batch

Evidence integration (CTD, PubTator, DGIdb, LINCS, KG paths/HAKE) retains your original logic.
Only GPT calling strategy changed to reduce API calls from N to 2 per drug.

Outputs:
- JSONL: one line per kept gene with all evidence fields
- For top-K genes, an additional "evidence_summary" field is attached
"""

import argparse
import csv
import json
import re
import numpy as np
import torch
from collections import defaultdict, OrderedDict
import requests
from urllib.parse import quote
import openai
import pickle
import sys
import os
import pathlib
from pathlib import Path
import time
import math
import random
from typing import List, Dict, Tuple, Set, Iterable, Optional
import pandas as pd

# -----------------------
# Paths for embeddings / dicts (same as your originals)
# -----------------------
entity_embedding_path = Path("HAKE_model/PrimeKGwE/entity_embedding.npy")
relation_embedding_path = Path("HAKE_model/PrimeKGwE/relation_embedding.npy")
relation_dict_path = Path("data/PrimeKG/relations.dict")
entity_dict = "data/PrimeKG/entities.dict"

entity_embed = np.load(entity_embedding_path)
relation_embed = np.load(relation_embedding_path)

MIN_PREFIX = 5  # Only treat as prefix match if shared prefix length >= 5

with open(relation_dict_path, "r") as f:
    relation_dict = dict(line.strip().split("\t")[::-1] for line in f)
    relation_dict = {k: int(v) for k, v in relation_dict.items()}

with open("pickle/PrimeKG/NER_ID_dict_cap_final.pickle", "rb") as f:
    name_to_idx = pickle.load(f)

with open("pickle/PrimeKG/biokdeid_type_map.pickle", "rb") as f:
    biokdeid_type_map = pickle.load(f)

with open("pickle/PrimeKG/DBRelations.no5.pickle", "rb") as f:
    DBRelations = pickle.load(f)

# ------------------------------------------------------------------------------------
# General utilities
# ------------------------------------------------------------------------------------

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # tolerate occasional stray text
                # print(f"[WARN] Bad JSON at {path}:{ln}: {e}")
                pass
    return rows

def write_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def infer_disease_from_lines_or_filename(input_path: str, lines: List[Dict]) -> str:
    for r in lines:
        for key in ("Disease", "disease", "DISEASE"):
            d = r.get(key)
            if isinstance(d, str) and d.strip():
                return d.strip()
    base = os.path.basename(input_path)
    m = re.match(r"^(.*?)\.disease2drug\.candidate\.jsonl$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return "UnknownDisease"

def normalize_gene_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    # remove spaces and uppercase
    x = re.sub(r"\s+", "", x).upper()
    return x

def normalize_drug_for_filename(x: str) -> str:
    x = (x or "UnknownDrug").strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^A-Za-z0-9._-]", "", x)
    return x or "UnknownDrug"

def uniq_preserve_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def filter_snippets(snips, keyword1, keyword2):
    if not snips:
        return []
    k1 = normalize_for_match(keyword1)
    k2 = normalize_for_match(keyword2)
    kept = []
    for s in snips:
        norm = normalize_for_match(s)
        if k1 in norm and k2 in norm:
            kept.append(s)
    return kept

def build_exact_name_index(name_to_idx: dict,
                           biokdeid_type_map: dict | None = None,
                           allowed_types: set[str] | None = None) -> dict[str, list[str]]:
    index = defaultdict(list)
    def keep_id(bid) -> bool:
        if not allowed_types or biokdeid_type_map is None:
            return True
        bid_str = str(bid)
        t = biokdeid_type_map.get(bid_str)
        if t is None and bid_str.isdigit():
            t = biokdeid_type_map.get(int(bid_str))
        return (t in allowed_types)
    for section, mapping in name_to_idx.items():
        if section.lower() == "id" or not isinstance(mapping, dict):
            continue
        for name, ids in mapping.items():
            key = normalize_for_match(name)
            if not key:
                continue
            ids_list = ids if isinstance(ids, list) else [ids]
            for bid in ids_list:
                if keep_id(bid):
                    bid_str = str(bid)
                    if bid_str not in index[key]:
                        index[key].append(bid_str)
    return index

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

def load_parameters(param_file):
    params = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split("\t")
                params[key] = value
    return params

def ask_gpt(messages, params, max_completion_tokens=4096, max_retries: int = 8):
    """
    Robust wrapper to call Azure OpenAI Chat Completions with retry.
    Retries on transient network errors and HTTP 429 (rate limit).
    Sleeps a random 1–10 seconds between attempts.

    Args:
        messages: list of chat messages
        params: dict containing AZURE_OPENAI_ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT_NAME
        max_completion_tokens: response token cap
        max_retries: maximum retry attempts (including the first try)
    """
    # Prepare Azure OpenAI client (outside retry loop is fine)
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

    # Import exception classes defensively (older SDKs may not have all)
    RateLimitError = getattr(openai, "RateLimitError", Exception)
    APIConnectionError = getattr(openai, "APIConnectionError", Exception)
    APIError = getattr(openai, "APIError", Exception)

    attempt = 0
    last_err = None

    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_completion_tokens
            )
            return response.choices[0].message.content

        except (RateLimitError, APIConnectionError, APIError) as e:
            # Check for HTTP status 429 on generic APIError as well
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            is_429 = (getattr(e, "code", None) == 429) or (status == 429)
            # Treat all three classes as transient; specifically mark 429
            transient = True

            attempt += 1
            last_err = e

            if attempt >= max_retries:
                break

            # Sleep random 1–10 seconds (per your requirement)
            sleep_sec = random.randint(1, 10)
            print(f"[ask_gpt][WARN] Attempt {attempt} failed ({'429' if is_429 else 'transient error'}). "
                  f"Sleeping {sleep_sec}s before retry...")
            time.sleep(sleep_sec)

        except Exception as e:
            # Unknown error: retry as transient once, then escalate if persists
            attempt += 1
            last_err = e

            if attempt >= max_retries:
                break

            sleep_sec = random.randint(1, 10)
            print(f"[ask_gpt][WARN] Attempt {attempt} failed (unexpected error: {type(e).__name__}). "
                  f"Sleeping {sleep_sec}s before retry...")
            time.sleep(sleep_sec)

    # If we’re here, all attempts failed
    raise RuntimeError(f"Failed to call Azure OpenAI after {max_retries} attempts: {last_err}")


def json_from_model_output(text: str):
    """Try to extract JSON from model output that may include fences or prose."""
    # Strip markdown fences
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first {...} or [...] block
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

# -----------------------
# PubTator cleaning
# -----------------------
def clean_pubtator_tags(snippets):
    cleaned = []
    for s in snippets:
        if not isinstance(s, str):
            continue
        s = re.sub(r"@(?:CHEMICAL|DISEASE|GENE|VARIANT|SPECIES|CELLLINE)_[^\s@]+", "", s)
        s = re.sub(r"@@@<m>(.*?)</m>@@@", r"\1", s)
        s = re.sub(r"@@@", r"", s)
        s = re.sub(r"<m>(.*?)</m>", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        cleaned.append(s)
    return cleaned

def normalize_drug_name(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip()).lower()

# -----------------------
# Inputs
# -----------------------
def load_candidates(jsonl_path: str):
    items: List[dict] = []
    name_index: Dict[str, List[int]] = defaultdict(list)
    kgid_index: Dict[str, List[int]] = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[!] Skip invalid JSONL at line {i}: {e}")
                continue
            gene_name = str(obj.get("Gene", "")).strip()
            gene_norm = normalize_gene_name(gene_name)
            obj["_norm_gene"] = gene_norm
            gene_kgid = str(obj.get("ID", "")).strip()
            items.append(obj)
            idx = len(items) - 1
            if gene_norm:
                name_index[gene_norm].append(idx)
            if gene_kgid:
                kgid_index[gene_kgid].append(idx)
    return items, name_index, kgid_index

def load_ctd_for_drug(tsv_path: str, drug_query_name: str):
    drug_q = normalize_drug_name(drug_query_name)
    ev_by_gene: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    drug_id_counter: Dict[str, int] = defaultdict(int)
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 5:
                continue
            drug_name, drug_id, gene_symbol, _gene_id, evidence_str = row[:5]
            if normalize_drug_name(drug_name) != drug_q:
                continue
            if drug_id and drug_id.upper() != "NULL":
                drug_id_counter[drug_id] += 1
            norm_gene = normalize_gene_name(gene_symbol)
            if not norm_gene:
                continue
            if evidence_str and evidence_str.strip().upper() != "NULL":
                parts = [p.strip() for p in evidence_str.split(";") if p.strip()]
                for p in parts:
                    if ":" in p:
                        ev_type, pmid = p.split(":", 1)
                        ev_type = ev_type.strip()
                        pmid = pmid.strip()
                    else:
                        ev_type = p.strip()
                        pmid = ""
                    if ev_type:
                        ev_by_gene[norm_gene].add((ev_type, pmid))
    inferred_drug_kgid = None
    if drug_id_counter:
        inferred_drug_kgid = max(drug_id_counter.items(), key=lambda x: x[1])[0]
    return ev_by_gene, inferred_drug_kgid

def collect_pubtator_gene_kgids_from_tsv(tsv_path: str, drug_kgid: str) -> Set[str]:
    if not drug_kgid:
        return set()
    allow: Set[str] = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 4)
            if len(parts) < 4:
                continue
            drug_id2 = parts[1].strip()
            gene_kgid = parts[3].strip()
            if drug_id2 == str(drug_kgid) and gene_kgid and gene_kgid.upper() != "NULL":
                allow.add(gene_kgid)
    return allow

def load_pubtator_from_tsv(tsv_path: str, drug_kgid: str):
    """
    Offline PubTator loader.
    Returns:
        Dict[gene_kgid] = {
            "text_hl": [...],
            "relation": {"type": str, "publications": int}
        }
    """
    result = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            # 假設格式：
            # drug_name, drug_kgid, gene_name, gene_kgid, relation_type, publications, snippet
            drug_id = parts[1].strip()
            gene_kgid = parts[3].strip()
            relation_type = parts[4].strip()
            publications = parts[5].strip()
            snippet = parts[6].strip() if len(parts) > 6 else ""

            if drug_id != str(drug_kgid):
                continue

            entry = result.setdefault(gene_kgid, {})

            if relation_type:
                entry["relation"] = {
                    "type": relation_type,
                    "publications": int(publications) if publications.isdigit() else 0
                }

            if snippet:
                entry.setdefault("text_hl", []).append(snippet)

    return result

def load_pubtator_for_drug_by_kgids(drug_name: str,
                                    gene_kgid_to_name: Dict[str, str],
                                    max_results: int = 5,
                                    session: requests.Session | None = None,
                                    allowed_gene_kgids: Set[str] | None = None):
    sess = session or requests.Session()
    base = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

    cid_cache: Dict[str, str | None] = {}
    
    def _first_concept_id(q: str,
                          max_retries: int = 5,
                          base_delay: float = 1.0) -> str | None:
        """
        Query PubTator autocomplete with retry/backoff + simple in-memory cache.

        - Cache key is the raw query string q.
        - Retries on HTTP 429 and transient RequestException.
        - Exponential backoff with small random jitter.
        """
        # 1) cache lookup
        if q in cid_cache:
            return cid_cache[q]

        for attempt in range(1, max_retries + 1):
            url = f"{base}/entity/autocomplete/?query={quote(q)}"
            try:
                r = sess.get(url, timeout=20)

                # 429: treat as rate limit
                if r.status_code == 429:
                    raise requests.exceptions.HTTPError(
                        f"429 Too Many Requests: {url}",
                        response=r
                    )

                r.raise_for_status()
                data = r.json()

                cid: str | None = None
                if isinstance(data, list) and data:
                    cid = data[0].get("_id")

                # 寫入 cache（即便是 None 也 cache，避免重打）
                cid_cache[q] = cid
                return cid

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    # 429: backoff + retry
                    if attempt >= max_retries:
                        print(f"[PubTator][_first_concept_id] 429 for '{q}', "
                              f"reached max_retries={max_retries}, giving up.")
                        cid_cache[q] = None
                        return None
                    delay = base_delay * (2 ** (attempt - 1)) + random.random()
                    print(f"[PubTator][_first_concept_id] 429 for '{q}', "
                          f"attempt {attempt}/{max_retries}, sleeping {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    # 其他 HTTP error：不 retry，直接放棄這個 query
                    print(f"[PubTator][_first_concept_id] HTTPError for '{q}': "
                          f"status={status}, error={e}")
                    cid_cache[q] = None
                    return None

            except requests.exceptions.RequestException as e:
                # 網路層 transient error：可 retry
                if attempt >= max_retries:
                    print(f"[PubTator][_first_concept_id] RequestException for '{q}' "
                          f"after {max_retries} attempts: {e}")
                    cid_cache[q] = None
                    return None
                delay = base_delay * (2 ** (attempt - 1)) + random.random()
                print(f"[PubTator][_first_concept_id] RequestException for '{q}', "
                      f"attempt {attempt}/{max_retries}, sleeping {delay:.1f}s...")
                time.sleep(delay)
                continue

        # 理論上不會到這裡；防禦性 return
        cid_cache[q] = None
        return None

    def _search_relations_snippets(gene_cid: str, chem_cid: str) -> list[str]:
        text = f"relations:ANY|{gene_cid}|{chem_cid}"
        url = f"{base}/search/?text={quote(text)}"

        try:
            r = sess.get(url, timeout=(10, 120))  # (connect timeout, read timeout)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.ReadTimeout:
            print(f"[PubTator][WARN] ReadTimeout for {gene_cid} - {chem_cid}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"[PubTator][WARN] Request error for {gene_cid} - {chem_cid}: {e}")
            return []
        except Exception as e:
            print(f"[PubTator][WARN] Unexpected error: {e}")
            return []

        hits = data.get("results", []) if isinstance(data, dict) else []
        out = []
        for h in hits[:max_results]:
            th = h.get("text_hl")
            if th:
                out.append(th)
        return clean_pubtator_tags(out)

    def _search_comention_snippets(gene_cid: str, chem_cid: str, gene_name_q: str = "", chem_name_q: str = "") -> list[str]:
        """
        Try co-mention search with concept IDs; on 400/empty results, fall back to plain-text names.
        This avoids 400 errors when IDs include special characters like (+) or parentheses.
        """
        attempts = []

        # 1) raw concept IDs
        attempts.append(f"{gene_cid} and {chem_cid}")

        # 2) stripped IDs without leading "@"
        gcid_stripped = re.sub(r"^@", "", str(gene_cid or ""))
        ccid_stripped = re.sub(r"^@", "", str(chem_cid or ""))
        if gcid_stripped and ccid_stripped:
            attempts.append(f"{gcid_stripped} and {ccid_stripped}")

        # 3) quoted plain-text names
        if gene_name_q and chem_name_q:
            attempts.append(f"\"{gene_name_q}\" and \"{chem_name_q}\"")
            attempts.append(f"{gene_name_q} and {chem_name_q}")

        for q in attempts:
            try:
                url = f"{base}/search/?text={quote(q)}"
                r = sess.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                hits = data.get("hits", []) if isinstance(data, dict) else []
                out = []
                for h in hits[:max_results]:
                    th = h.get("text_hl")
                    if not th:
                        continue
                    if isinstance(th, list):
                        out.extend([t for t in th if t])
                    elif isinstance(th, str):
                        out.append(th)
                cleaned = clean_pubtator_tags(out)
                if cleaned:
                    return cleaned
            except requests.exceptions.HTTPError as e:
                # On 400 Bad Request (e.g., due to special characters), continue to next attempt
                if getattr(e.response, "status_code", None) == 400:
                    continue
                else:
                    # For other HTTP errors, also try the next attempt
                    continue
            except Exception:
                # Network/parse issue; try next attempt
                continue

        # If all attempts fail or no snippets found
        return []

    def _relations_first(gene_cid: str, chem_cid: str):
        url = f"{base}/relations?e1={quote(gene_cid)}&e2={quote(chem_cid)}"
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 404:
                return None
            # May raise for 4xx/5xx; we'll handle 400 (and others) below
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError as e:
            # Explicitly skip cases that trigger Bad Request (e.g., concept ids with special characters)
            status = getattr(e.response, "status_code", None)
            if status == 400:
                return None
            # For any other HTTP error, also skip this pair silently
            return None
        except Exception:
            # Network/parse error: skip this pair
            return None

        if not isinstance(data, list) or not data:
            return None
        data = sorted(
            [d for d in data if isinstance(d, dict) and "type" in d and "publiculations" in d],
            key=lambda x: x.get("publications", 0),
            reverse=True
        )
        if not data:
            return None
        top = data[0]
        return {"type": top.get("type"), "publications": int(top.get("publications", 0))}

    drug_cid = _first_concept_id(drug_name)
    if not drug_cid:
        print(f"[PubTator] No concept id found for drug: {drug_name}")
        return {}

    results: Dict[str, dict] = {}
    items_iter = gene_kgid_to_name.items()
    if allowed_gene_kgids:
        items_iter = ((gid, gene_kgid_to_name[gid]) for gid in allowed_gene_kgids if gid in gene_kgid_to_name)

    for gene_kgid, gene_name in items_iter:
        if not gene_name:
            continue
        time.sleep(0.2)
        gene_cid = _first_concept_id(str(gene_name))
        if not gene_cid:
            continue
        snippets = _search_relations_snippets(gene_cid, drug_cid)
        if not snippets:
            snippets = _search_comention_snippets(gene_cid, drug_cid, gene_name_q=str(gene_name), chem_name_q=str(drug_name))
        rel = _relations_first(gene_cid, drug_cid)
        if snippets or rel:
            payload = {}
            if snippets:
                payload["text_hl"] = snippets
            if rel:
                payload["relation"] = rel
            results[gene_kgid] = payload
    return results

def load_dgidb_for_drug_by_kgids(tsv_path: str, drug_kgid: str):
    dgidb_by_gene_kgid: Dict[str, List[dict]] = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            drug_id2 = parts[1].strip()
            gene_kgid = parts[3].strip()
            ev_str = parts[4].strip()
            if drug_id2 != str(drug_kgid):
                continue
            if not gene_kgid or not ev_str:
                continue
            ev_type = None
            confidence_score = None
            approved = False
            anti_neoplastic = False
            immunotherapy = False
            tokens = [t.strip() for t in ev_str.split(";") if t.strip()]
            for tok in tokens:
                if ":" in tok:
                    t, val = tok.split(":", 1)
                    t = t.strip()
                    val = val.strip()
                    if ev_type is None:
                        ev_type = t if t else None
                        confidence_score = val if val else None
                else:
                    low = tok.lower()
                    if low == "approved":
                        approved = True
                    elif low == "anti_neoplastic":
                        anti_neoplastic = True
                    elif low == "immunotherapy":
                        immunotherapy = True
            if ev_type is None:
                continue
            dgidb_by_gene_kgid[gene_kgid].append({
                "relation type": ev_type,
                "confidence_score": confidence_score,
                "FDA-approved": True if approved else False,
                "anti_neoplastic": True if anti_neoplastic else False,
                "immunotherapy": True if immunotherapy else False
            })
    return dgidb_by_gene_kgid

def load_lincs_for_drug_by_kgids(tsv_path: str, drug_kgid: str):
    lincs_by_gene_kgid: Dict[str, dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        required = ["Chemical_kg_id", "Gene_kg_id", "avg_coef", "direction", "UpEvidence", "DownEvidence"]
        if not all(k in col for k in required):
            raise ValueError("LINCS header missing required columns: " + ", ".join(required))
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            chem_kgid = parts[col["Chemical_kg_id"]].strip()
            gene_kgid = parts[col["Gene_kg_id"]].strip()
            avg_coef = parts[col["avg_coef"]].strip()
            direction = parts[col["direction"]].strip().lower()
            up_str = parts[col["UpEvidence"]].strip()
            down_str = parts[col["DownEvidence"]].strip()
            if chem_kgid != str(drug_kgid):
                continue
            if not gene_kgid:
                continue
            entry = lincs_by_gene_kgid.get(gene_kgid, {
                "avg_coef": "",
                "direction": "",
                "UpEvidence":   {"_list": []},
                "DownEvidence": {"_list": []},
            })
            if avg_coef:
                entry["avg_coef"] = avg_coef
            if direction:
                entry["direction"] = direction
            def parse_cell(cell: str) -> List[str]:
                if not cell:
                    return []
                return [t for t in (s.strip() for s in cell.split(";")) if t]
            if direction == "up":
                entry["UpEvidence"]["_list"].extend(parse_cell(up_str))
            elif direction == "down":
                entry["DownEvidence"]["_list"].extend(parse_cell(down_str))
            lincs_by_gene_kgid[gene_kgid] = entry
    for gid, entry in lincs_by_gene_kgid.items():
        up_list = entry["UpEvidence"]["_list"]
        up_list = list(OrderedDict.fromkeys(up_list)) if up_list else []
        if up_list:
            entry["UpEvidence"] = {"records": "|".join(up_list), "#records": len(up_list)}
        else:
            entry.pop("UpEvidence", None)
        down_list = entry["DownEvidence"]["_list"]
        down_list = list(OrderedDict.fromkeys(down_list)) if down_list else []
        if down_list:
            entry["DownEvidence"] = {"records": "|".join(down_list), "#records": len(down_list)}
        else:
            entry.pop("DownEvidence", None)
    return lincs_by_gene_kgid

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

def infer_up_down_from_evidence(evidence_dict):
    lincs = evidence_dict.get("LINCS database")
    if isinstance(lincs, dict):
        dir_val = str(lincs.get("direction", "")).strip().lower()
        if dir_val in ("up", "down"):
            return {"direction": dir_val, "source": "LINCS"}
    dgidb = evidence_dict.get("DGIdb database")
    if isinstance(dgidb, list) and dgidb:
        up_ct, down_ct = 0, 0
        for rec in dgidb:
            t = str(rec.get("relation type", "")).strip().lower()
            if t in ("activator", "agonist"):
                up_ct += 1
            elif t == "inhibitor":
                down_ct += 1
        if up_ct or down_ct:
            if up_ct > down_ct:
                return {"direction": "up", "source": "DGIdb"}
            elif down_ct > up_ct:
                return {"direction": "down", "source": "DGIdb"}
            else:
                return {"direction": None, "source": "DGIdb"}
    ctd = evidence_dict.get("CTDbase")
    if isinstance(ctd, list) and ctd:
        up_ct, down_ct = 0, 0
        for rec in ctd:
            t = str(rec.get("type", "")).strip().lower()
            if t == "increases-expression":
                up_ct += 1
            elif t == "decreases-expression":
                down_ct += 1
        if up_ct or down_ct:
            if up_ct > down_ct:
                return {"direction": "up", "source": "CTD"}
            elif down_ct > up_ct:
                return {"direction": "down", "source": "CTD"}
            else:
                return {"direction": None, "source": "CTD"}
    pt = evidence_dict.get("PubTator (Machine) recognized relation")
    if isinstance(pt, str) and pt:
        t_match = re.search(r"\b(positive_correlate|negative_correlate)\b", pt, re.I)
        n_match = re.search(r"\bin\s+(\d+)\s+publications\b", pt, re.I)
        if t_match and n_match:
            n = int(n_match.group(1))
            if n >= 10:
                t = t_match.group(1).lower()
                return {"direction": "up" if t == "positive_correlate" else "down", "source": "PubTator"}
    return {"direction": None, "source": None}

# -----------------------
# GPT 2-stage helpers
# -----------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(template_str: str, **kw):
    out = template_str
    for k, v in kw.items():
        out = out.replace(f"[{k}]", v)
    return out

def minimize_for_prompt(entry: dict, max_paths=3, max_snips=3) -> dict:
    """Produce a compact version of out_obj for prompt usage."""
    d = {
        "Gene": entry.get("Gene", ""),
        "Disease": entry.get("Disease", ""),
        "Drug": entry.get("Drug", ""),
        "GeneID": entry.get("GeneID", ""),
    }
    # Keep limited KG paths
    kgp = entry.get("Drug_Gene_KG_evidence", []) or []
    if kgp:
        d["Drug_Gene_KG_evidence_top"] = kgp[:max_paths]
    # Keep limited disease snippets
    sn = entry.get("Gene_to_Disease_snippet", []) or []
    if sn:
        d["Gene_to_Disease_snippet_top"] = sn[:max_snips]
    # Keep evidence containers but they may be large; keep as-is
    ev = entry.get("Drug_to_Gene_evidence", {})
    if ev:
        d["Drug_to_Gene_evidence"] = ev
    # Optional mechanism
    mech = entry.get("Gene_Disease_Mechanism")
    if mech:
        d["Gene_Disease_Mechanism"] = mech
    return d

def call_gpt_for_topk(all_gene_evidence: List[dict], drug_name: str, disease_name: str,
                      prompt_topk_str: str, params: dict, k: int,
                      retries: int = 5, sleep_seconds: int = 5,
                      json_input_str: str | None = None) -> List[str]:
    """
    Added `json_input_str` to inject disease-detail JSON into the [JSON_Input] placeholder of the ranking prompt.
    """
    compact = [minimize_for_prompt(e) for e in all_gene_evidence]
    input_text = json.dumps(compact, ensure_ascii=False, indent=2)

    # Provide JSON_Input to the template if available; empty string otherwise.
    prompt = render_prompt(
        prompt_topk_str,
        Drug=drug_name,
        Disease=disease_name,
        k=str(k),
        input=input_text,
        JSON_Input=(json_input_str or "")
    )
    
    # Retry loop around ask_gpt
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = ask_gpt(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}],
                params,
                max_completion_tokens=4096
            )
            break  # success
        except Exception as e:
            last_err = e
            sys.stderr.write(f"[WARN] call_gpt_for_topk attempt {attempt}/{retries} failed: {e}\n")
            if attempt < retries:
                time.sleep(sleep_seconds)
            else:
                # On final failure, re-raise to preserve original behavior after retries
                raise

    data = json_from_model_output(resp)
    if not data:
        print("[WARN] TopK GPT response not valid JSON; fallback to empty list", file=sys.stderr)
        return []

    genes = []
    if isinstance(data, dict):
        if "top_genes" in data:
            for g in data.get("top_genes", []):
                if isinstance(g, dict) and "Gene" in g:
                    genes.append(g["Gene"])
        elif "Genes" in data and isinstance(data["Genes"], list):
            for g in data["Genes"]:
                if isinstance(g, str):
                    genes.append(g.strip())
    elif isinstance(data, list):
        for g in data:
            if isinstance(g, dict):
                if "Gene" in g:
                    genes.append(g["Gene"])
                elif "Genes" in g and isinstance(g["Genes"], list):
                    for item in g["Genes"]:
                        if isinstance(item, str):
                            genes.append(item.strip())
            elif isinstance(g, str):
                genes.append(g.strip())

    # 去重、保留順序並取前 k 個
    seen = set()
    uniq = []
    for g in genes:
        if g not in seen:
            uniq.append(g)
            seen.add(g)
    
    return uniq[:k]

def call_gpt_for_summaries(top_gene_evidence: List[dict], drug_name: str, disease_name: str,
                           prompt_summary_str: str, params: dict,
                           retries: int = 5, sleep_seconds: int = 5) -> Dict[str, str]:
    """
    Call GPT once to summarize evidence for top genes. On network/API failure,
    retry up to `retries` times with `sleep_seconds` pause between attempts.
    """
    compact = [minimize_for_prompt(e) for e in top_gene_evidence]
    input_text = json.dumps(compact, ensure_ascii=False, indent=2)
    prompt = render_prompt(prompt_summary_str, Drug=drug_name, Disease=disease_name, input=input_text)

    # Retry loop around ask_gpt
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = ask_gpt(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}],
                params,
                max_completion_tokens=4096
            )
            break  # success
        except Exception as e:
            last_err = e
            sys.stderr.write(f"[WARN] call_gpt_for_summaries attempt {attempt}/{retries} failed: {e}\n")
            if attempt < retries:
                time.sleep(sleep_seconds)
            else:
                # On final failure, re-raise to preserve original behavior after retries
                raise

    data = json_from_model_output(resp)
    if not data:
        print("[WARN] Summary GPT response not valid JSON; returning empty", file=sys.stderr)
        return {}
    # Accept list of {"Gene": "...", "evidence_summary": "..."}
    out = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            g = item.get("Gene")
            s = item.get("evidence_summary")
            if g and s:
                out[g] = s
    elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            g = item.get("Gene")
            s = item.get("evidence_summary")
            if g and s:
                out[g] = s
    return out

def _common_prefix_len(a: str, b: str) -> int:
    """Return the number of identical leading characters shared by a and b."""
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Integrate CTD/PubTator/DGIdb/LINCS evidence; use GPT to rank genes by association with [Drug]-[Disease]; collect evidence in ranked order until Top-K have evidence; then summarize once and exit."
    )
    ap.add_argument("--input", required=True, help="output/disease2drug/[Disease].disease2drug.candidate.jsonl")
    ap.add_argument("--disease2gene_candidate_json", required=True)
    ap.add_argument("--ctd_tsv", required=True)
    ap.add_argument("--pubtator_tsv", required=True)
    ap.add_argument("--dgidb_tsv", required=True)
    ap.add_argument("--lincs_tsv", required=True)
    ap.add_argument("--param_file", default="parameter.gpt4o.txt")
    ap.add_argument("--prompt_topk", default="prompts/3.GPT4DrugGene.Ranking.txt",
                    help="Prompt template with placeholders [Drug], [Disease], [input], [JSON_Input] (used here to RANK all genes)")
    ap.add_argument("--prompt_summary", default="prompts/3.GPT4DrugGene.SummarizingEvidence.txt",
                    help="Prompt template with placeholders [Drug], [Disease], [input]")
    ap.add_argument("--disease_detail_json", required=False,
                    help="Path to disease-detail JSON (as raw text) to inject into [JSON_Input] in the ranking prompt")

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--output_folder")
    args = ap.parse_args()

    # Read disease2drug candidate file
    lines = read_jsonl(args.input)
    if not lines:
        raise RuntimeError("No records found in --input: {}".format(args.input))
    disease = infer_disease_from_lines_or_filename(args.input, lines)

    # Collect ALL drugs
    drugs = []
    for r in lines:
        for key in ("Drug", "drug", "DRUG", "candidate_drug"):
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                drugs.append(v.strip())
                break
    uniq_drugs = uniq_preserve_order(drugs)
    if not uniq_drugs:
        raise RuntimeError("No 'Drug' field found in --input")

    # ensure output folder
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    # load once
    params = load_parameters(args.param_file)
    id_to_official = load_name_map("pickle/PrimeKG/NER_ID_dict_cap_final.pickle")

    for i, drug_name in enumerate(uniq_drugs, 1):
        safe_drug = normalize_drug_for_filename(drug_name)
        out_path = os.path.join(args.output_folder or ".", f"{normalize_for_match(disease)}.{drug_name}.jsonl")

        #print(args.output_folder,f"{normalize_for_match(disease)}.{drug_name}.jsonl") 
        #exit(1)
        if os.path.exists(out_path):
            print(f"[SKIP] Output file already exists for {drug_name}: {out_path}")
        else:
            print("[INFO] ({}/{}) Processing drug: {}".format(i, len(uniq_drugs), drug_name))

            # 1) load candidates
            print("1) disease2gene_candidate_json")
            items, name_index, kgid_index = load_candidates(args.disease2gene_candidate_json)
            disease_name = (items[0].get("Disease") if items else None) or disease

            # build helper maps for quick lookups
            # prefer case-insensitive gene key
            by_gene_ci = {}
            for it in items:
                g = (it.get("Gene") or "").strip()
                if g:
                    by_gene_ci.setdefault(g.upper(), it)

            ## 2) GPT ranking for ALL genes (no pre-filter)
            print("2) GPT ranking for all genes")
            prompt_topk_str = load_text(args.prompt_topk)

            # Optional: load disease-detail JSON to populate [JSON_Input] in the ranking prompt.
            case_json_str = ""
            if args.disease_detail_json:
                # Read the file verbatim so the prompt gets the exact JSON the user supplied.
                with open(args.disease_detail_json, "r", encoding="utf-8") as f:
                    case_json_str = f.read().strip()

            cand_min = [
                {
                    "Gene": it.get("Gene", ""),
                    "Disease": it.get("Disease", disease_name),
                    "Drug": drug_name,
                    "GeneID": str(it.get("ID", "")).strip(),
                }
                for it in items if it.get("Gene")
            ]
            if not cand_min:
                print("[WARN] No candidate genes found for {}".format(drug_name))
                continue

            # ask GPT to return an ordered list; we pass k=len(cand_min) to get a full ranking
            ranked_genes = call_gpt_for_topk(
                cand_min, drug_name, disease_name, prompt_topk_str, params,
                k=len(cand_min), json_input_str=case_json_str
            )
            #print("[Rank] First 20 genes:", ranked_genes[:20])

            # 3) CTD
            print("3) CTD (name-based parsing)")
            ctd_ev_by_gene, inferred_drug_kgid = load_ctd_for_drug(args.ctd_tsv, drug_name)

            # 4) Resolve drug_kgid
            print("4) drug_kgid for PubTator/DGIdb/LINCS")
            drug_kgid = inferred_drug_kgid
            if not drug_kgid:
                exact_idx = build_exact_name_index(name_to_idx, biokdeid_type_map, allowed_types={"Chemical"})
                q = normalize_for_match(drug_name)
                ids = exact_idx.get(q)
                if ids:
                    drug_kgid = ids[0]
                    print(f"[i] Found drug_kgid by exact name: {drug_kgid}")
                else:
                    candidates = []
                    # Only consider true prefix matches, and require a minimum shared prefix length.
                    for key, id_list in exact_idx.items():
                        # true prefix check (avoid substring hits like "as" in "dasatinib")
                        if not (key.startswith(q) or q.startswith(key)):
                            continue

                        shared_prefix_len = _common_prefix_len(key, q)
                        if shared_prefix_len < MIN_PREFIX:
                            # below threshold: skip this candidate entirely
                            continue

                        # Secondary tie-breaker: prefer closer lengths
                        len_penalty = abs(len(key) - len(q)) / max(len(key), 1)

                        # Lower score is better; with threshold enforced, prefix_penalty is always 0 here
                        score = (0, len_penalty)
                        candidates.append((score, id_list[0], key))

                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        drug_kgid = candidates[0][1]
                        key_used = candidates[0][2]
                        print(f"[i] Found drug_kgid by prefix '{key_used}': {drug_kgid}")
                    else:
                        print(f"[i] No suitable drug_kgid by prefix ≥{MIN_PREFIX}; will skip KGID-dependent steps.")
            if not drug_kgid:
                print("[i] No drug_kgid available; PubTator/DGIdb/LINCS steps that require KGID will be skipped.")

            # 5) DGIdb and LINCS once per drug (maps from gene kgid -> evidence)
            print("5) DGIdb")
            dgidb_by_gene_kgid: Dict[str, List[dict]] = {}
            if drug_kgid:
                dgidb_by_gene_kgid = load_dgidb_for_drug_by_kgids(args.dgidb_tsv, drug_kgid=str(drug_kgid))

            print("6) LINCS")
            lincs_by_gene_kgid: Dict[str, dict] = {}
            if drug_kgid:
                lincs_by_gene_kgid = load_lincs_for_drug_by_kgids(args.lincs_tsv, drug_kgid=str(drug_kgid))

            # 7) iterate ranked genes; collect evidence until Top-K have evidence, then stop
            print("7) Collect evidence in ranked order until K reached")
            nodeid_to_index = {str(v["node_id"]): v.index for v in DBRelations.vs}
            selected_gene_objs: List[dict] = []
            selected_gene_names_norm: set = set()

            # we will lazily fetch PubTator per small batches to reduce overhead
            session = None  # can pass a requests.Session if you prefer
            for g in ranked_genes:
                if len(selected_gene_objs) >= args.topk:
                    break
                gkey = (g or "").strip().upper()
                if not gkey or gkey not in by_gene_ci:
                    continue
                row = by_gene_ci[gkey]

                gene_name = row.get("Gene", "")
                gene_kgid = str(row.get("ID", "")).strip()
                norm_gene = row.get("_norm_gene", "")

                # PubTator (fetch only for this gene if needed)
                pt_by_gene_kgid: Dict[str, dict] = {}
                if drug_kgid and gene_kgid:
                    allowed_gene_kgids = {gene_kgid}
                    if drug_kgid:
                        pt_by_gene_kgid = load_pubtator_from_tsv(
                            args.pubtator_tsv,
                            drug_kgid=str(drug_kgid)
                        )
                    else:
                        pt_by_gene_kgid = {}
                     
                # CTD
                ctd_clean = []
                if norm_gene in ctd_ev_by_gene:
                    ctd_pairs = sorted(ctd_ev_by_gene[norm_gene])  # (type, pmid)
                    ctd_clean = [{"type": t, "pmid": p} for (t, p) in ctd_pairs if t]

                # PubTator fields
                pubtator_snips = None
                pubtator_rel = ""
                if gene_kgid and gene_kgid in pt_by_gene_kgid:
                    rec = pt_by_gene_kgid[gene_kgid]
                    snips = rec.get("text_hl", [])
                    if snips:
                        snips = filter_snippets(snips, gene_name, drug_name)
                        if snips:
                            pubtator_snips = snips
                    rel = rec.get("relation")
                    if isinstance(rel, dict) and rel.get("type"):
                        pubs = rel.get("publications", 0)
                        try:
                            pubs = int(pubs)
                        except Exception:
                            pass
                        pubtator_rel = f'{drug_name} {rel.get("type")} {gene_name} in {pubs} publications'

                # DGIdb
                dgidb_clean = []
                if gene_kgid and gene_kgid in dgidb_by_gene_kgid:
                    dgidb_clean = dgidb_by_gene_kgid[gene_kgid]

                # LINCS
                lincs_clean = None
                if gene_kgid and gene_kgid in lincs_by_gene_kgid:
                    lincs_clean = lincs_by_gene_kgid[gene_kgid]

                has_any = bool(ctd_clean or pubtator_snips or pubtator_rel or dgidb_clean or lincs_clean or row.get("KG_evidence"))
                if not has_any:
                    continue
                
                # Clean KG_evidence strings by removing [score=...]
                raw_evidence = row.get("KG_evidence", [])
                cleaned_evidence = [re.sub(r"\s*\[score=[^\]]+\]", "", ev) for ev in raw_evidence]

                out_obj = {
                    "Gene": gene_name,
                    "Disease": row.get("Disease", "") or disease_name,
                    "Drug": drug_name,
                    "GeneID": gene_kgid,
                    "Gene_to_Disease_KG_evidence": cleaned_evidence,
                }
                filtered_snips = filter_snippets(row.get("snippet", []), gene_name, row.get("Disease", ""))
                if filtered_snips:
                    out_obj["Gene_to_Disease_snippet"] = filtered_snips
                mechanism = str(row.get("Mechanism", "")).strip()
                if mechanism and not mechanism.startswith("HAKE_Rank:"):
                    out_obj["Gene_Disease_Mechanism"] = mechanism

                # KG shortest paths for Drug->Gene
                paths = []
                if drug_kgid and gene_kgid and (drug_kgid in nodeid_to_index) and (gene_kgid in nodeid_to_index):
                    src_idx = nodeid_to_index[drug_kgid]
                    dst_idx = nodeid_to_index[gene_kgid]
                    paths = DBRelations.get_all_shortest_paths(src_idx, to=dst_idx, mode="OUT")

                if not paths:
                    out_obj["Drug_Gene_KG_evidence"] = []
                else:
                    scored_paths = []
                    for path in paths:
                        parts = []
                        for j in range(len(path) - 1):
                            u_name = DBRelations.vs[path[j]]["name"]
                            rel_id = DBRelations.get_eid(path[j], path[j + 1], directed=True, error=False)
                            rel = DBRelations.es[rel_id]["relationship_type"] if rel_id != -1 else "?"
                            parts.append(u_name)
                            parts.append(f"-({rel})->")
                        parts.append(DBRelations.vs[path[-1]]["name"])
                        path_str = " ".join(parts)
                        score = total_hake_score(DBRelations, path, entity_embed, relation_embed, relation_dict)
                        scored_paths.append((score, path_str))
                    scored_paths.sort(reverse=True, key=lambda x: x[0])
                    top10_paths = [f"{p} [score={s:.4f}]" for s, p in scored_paths[:10]]
                    out_obj["Drug_Gene_KG_evidence"] = [
                        convert_path_ids_to_names(p.split(" [score=")[0], id_to_official)
                        for p in top10_paths
                    ]

                evidence = {}
                if ctd_clean:
                    evidence["CTDbase"] = ctd_clean
                if pubtator_snips:
                    evidence["PubTator snippets"] = pubtator_snips
                if pubtator_rel:
                    evidence["PubTator (Machine) recognized relation"] = pubtator_rel
                if dgidb_clean:
                    evidence["DGIdb database"] = dgidb_clean
                if lincs_clean:
                    evidence["LINCS database"] = lincs_clean
                if evidence:
                    out_obj["Drug_to_Gene_evidence"] = evidence

                # accept this gene
                selected_gene_objs.append(out_obj)
                selected_gene_names_norm.add(gkey)

            # 8) Summarize Top-K (or fewer if not enough) in one call, then write and EXIT outer drug loop
            print("8) Summarize and write output, then exit drug loop")
            if not selected_gene_objs:
                print(f"[WARN] No genes with evidence found for {drug_name}. Writing empty file.")

            # keep original GPT rank order
            order_norm = { (g or "").strip().upper(): idx for idx, g in enumerate(ranked_genes) }
            final_genes = [g for g in selected_gene_objs]
            final_genes.sort(key=lambda x: order_norm.get((x.get("Gene","") or "").strip().upper(), 10**9))

            # trim to Top-K
            if len(final_genes) > args.topk:
                final_genes = final_genes[:args.topk]

            # call summary
            prompt_summary_str = load_text(args.prompt_summary)
            summaries_map = {}
            if final_genes:
                summaries_map = call_gpt_for_summaries(
                    final_genes, drug_name, disease_name, prompt_summary_str, params
                )

            # write
            safe_disease = normalize_drug_for_filename(disease_name)
            safe_drug = normalize_drug_for_filename(drug_name)
            out_path = os.path.join(args.output_folder or ".", f"{safe_disease}.{safe_drug}.jsonl")
            n_out = 0
            with open(out_path, "w", encoding="utf-8") as fout:
                for g in final_genes:
                    key = (g.get("Gene","") or "").strip()
                    gsum = summaries_map.get(key) or summaries_map.get(key.upper()) or summaries_map.get(key.lower())
                    if gsum:
                        g["evidence_summary"] = gsum
                    fout.write(json.dumps(g, ensure_ascii=False) + "\n")
                    n_out += 1

            if len(final_genes) < args.topk:
                print(f"[WARN] Only {len(final_genes)} genes with evidence available (requested K={args.topk}).")
            print(f"Done. Wrote {n_out} rows to {out_path}.")


if __name__ == "__main__":
    main()



"""
python 3.DrugKLM.disease2drug2gene.evidence.py \
  --input output/disease2drug/melanoma.disease2drug.candidate.jsonl \
  --disease2gene_candidate_json output/disease2gene/melanoma.disease2gene.candidate.jsonl \
  --ctd_tsv DB/DrugGeneRelationEvdience.CTD.tsv \
  --pubtator_tsv DB/DrugGeneRelationEvdience.PubTator3.tsv \
  --dgidb_tsv DB/DrugGeneRelationEvdience.dgidb.drug2genes.tsv \
  --lincs_tsv DB/DrugGeneRelationEvdience.LINCS.cp_coeff.tsv \
  --prompt_topk prompts/3.GPT4DrugGene.Ranking.txt \
  --disease_detail_json input/melanoma.json \
  --output_folder output/drug2gene/melanoma

python 3.DrugKLM.disease2drug2gene.evidence.py \
  --input output/disease2drug/mcrc.disease2drug.candidate.jsonl \
  --disease2gene_candidate_json output/disease2gene/mcrc.disease2gene.candidate.jsonl \
  --ctd_tsv DB/DrugGeneRelationEvdience.CTD.tsv \
  --pubtator_tsv DB/DrugGeneRelationEvdience.PubTator3.tsv \
  --dgidb_tsv DB/DrugGeneRelationEvdience.dgidb.drug2genes.tsv \
  --lincs_tsv DB/DrugGeneRelationEvdience.LINCS.cp_coeff.tsv \
  --prompt_topk prompts/3.GPT4DrugGene.Ranking.txt \
  --disease_detail_json input/mCRC.json \
  --output_folder output/drug2gene/mcrc
"""
