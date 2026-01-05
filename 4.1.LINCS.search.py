#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.1.LINCS.search.py (patched)

Output schema (exactly like previous version):
  Signature, LN_IC50, IC50, CellLine, Drug, Direction, Genes

Enhancements in this patched version:
- Robust disease matching: stop-term cleaning, bidirectional substring, simple synonyms (breast↔mammary, cancer↔carcinoma)
- Preserves AND matching for variants and disease-only fallback behavior
- Keeps prior CLI and output contract

CLI examples:
  python 4.1.LINCS.search.py \
    --input_json input/melanoma.json \
    --mapping_jsonl DB/cell2disease.merge.LINCS.variants.jsonl \
    --extra_jsonl extras.jsonl \
    --gmt_folder DB/LINCS/gmt \
    --gdsc_file2 DB/GDSC/GDSC2_fitted_dose_response_27Oct23.csv \
    --gdsc_file1 DB/GDSC/GDSC1_fitted_dose_response_27Oct23.csv \
    --ccle_file  DB/CCLE/DrugResponse.txt \
    --output output/LINCS/LINCS.search.melanoma.cell2gene.tsv

Also supports legacy --disease (no variants) if --input_json not provided.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Set, Optional

import pandas as pd

# -----------------------------
# Normalization & utils
# -----------------------------
EMPTY_TOKENS = {"", "-", "NA", "N/A", "nan", "NaN"}

def split_tokens(s: str) -> List[str]:
    if s is None:
        return []
    parts = re.split(r"[|;]", str(s))
    return [p.strip() for p in parts if p and p.strip() and p.strip() not in EMPTY_TOKENS]

def norm_cell(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()

def normalize_disease_for_match(d: str) -> str:
    # kept for backward compatibility (not used in matching anymore)
    return (d or "").lower()

def normalize_variant_token(tok: str) -> str:
    """
    Partial-match-friendly normalization:
    - remove spaces
    - drop p./c./g./r. prefixes
    - uppercase
    - convert 3-letter AA to 1-letter (VAL600GLU → V600E)
    """
    import re as _re
    AA3_TO_1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
        "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
        "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
    }
    def aa3_to_aa1(s: str) -> str:
        pat = _re.compile(r"([A-Z]{3})(\d+)([A-Z]{3})")
        def _rep(m):
            a1, pos, a2 = m.group(1), m.group(2), m.group(3)
            if a1 in AA3_TO_1 and a2 in AA3_TO_1:
                return f"{AA3_TO_1[a1]}{pos}{AA3_TO_1[a2]}"
            return m.group(0)
        return pat.sub(_rep, s)
    t = (tok or "").strip().replace(" ", "").upper()
    t = _re.sub(r"^[PCGR]\.", "", t)  # drop leading p./c./g./r.
    return aa3_to_aa1(t)

def ln_to_ic50(ln_value):
    try:
        return math.exp(float(ln_value))
    except (ValueError, TypeError):
        return ""

# -----------------------------
# Disease matching helpers
# -----------------------------
STOP_TERMS = ["cancer", "carcinoma", "neoplasm", "tumor", "malignancy", "malignant", "disease", "urinary"]

def clean_disease_term(disease: str) -> str:
    """Remove generic disease suffix terms and normalize spacing/symbols."""
    disease = (disease or "").lower()
    for term in STOP_TERMS:
        disease = re.sub(rf"\b{term}\b", "", disease, flags=re.IGNORECASE)
    disease = re.sub(r"[^a-z0-9\s]", " ", disease)
    disease = re.sub(r"\s+", " ", disease).strip()
    return disease

def _norm_token_for_disease(s: str) -> str:
    # lower + remove non-alphanum for robust substring checks
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _expand_disease_synonyms(q: str) -> List[str]:
    q = (q or "").lower().strip()
    toks: Set[str] = {q}
    if "breast" in q or "mammary" in q:
        toks.update(["breast", "mammary"]) 
    if "cancer" in q or "carcinoma" in q:
        toks.update(["cancer", "carcinoma"]) 
    return list(toks)

def disease_match_ok(disease_value: str, disease_query: str) -> bool:
    """
    Flexible disease matching:
    - Lowercase, strip, remove generic stop-terms (cancer/carcinoma/…)
    - Normalize (remove non-alphanum) then bidirectional substring
    - Simple synonym expansion (breast↔mammary, cancer↔carcinoma)
    """
    dv_raw = clean_disease_term((disease_value or "").strip().lower())
    dq_raw = clean_disease_term((disease_query or "").strip().lower())
    if not dq_raw:
        return True

    dv_n = _norm_token_for_disease(dv_raw)
    dq_n = _norm_token_for_disease(dq_raw)

    if dv_n and dq_n and (dq_n in dv_n or dv_n in dq_n):
        return True

    for syn in _expand_disease_synonyms(dq_raw):
        if syn and (syn in dv_raw):
            return True

    return False

# -----------------------------
# JSONL mapping & extras
# -----------------------------

def load_cell2disease_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping JSONL not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {ln}: {e}")
            rows.append({
                "CellLine": obj.get("Cellline", ""),
                "Disease":  obj.get("Disease", ""),
                "Variant":  obj.get("Variant", []),
            })
    return pd.DataFrame(rows)

def load_extra_jsonl(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["CellLine","Disease","Variant"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extra JSONL not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"[extra] Invalid JSON at line {ln}: {e}")
            rows.append({
                "CellLine": obj.get("Cellline", ""),
                "Disease":  obj.get("Disease", ""),
                "Variant":  obj.get("Variant", []),
            })
    return pd.DataFrame(rows)

# -----------------------------
# Variant criteria & matching (AND)
# -----------------------------

def parse_variant_filters(obj: Any) -> List[Dict[str, str]]:
    filters: List[Dict[str, str]] = []
    if isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            gene = str(it.get("gene", "")).strip()
            gene_id = str(it.get("gene_id", "")).strip()
            var = str(it.get("variant", "")).strip()
            vtype = str(it.get("variant_type", "")).strip().lower()
            if not var:
                continue
            filters.append({
                "gene": gene.upper(),
                "gene_id": gene_id,
                "variant": normalize_variant_token(var),
                "variant_type": vtype
            })
    return filters

def cell_has_all_variants(row: pd.Series, variant_filters: List[Dict[str, str]]) -> bool:
    entries = row.get("Variant", []) or []
    if not isinstance(entries, list):
        return False
    for crit in variant_filters:
        gene_need = crit["gene"]
        gene_id_need = crit["gene_id"]
        var_need = crit["variant"]
        this_ok = False
        for ent in entries:
            gsym = str(ent.get("GeneName", "")).upper()
            gid  = str(ent.get("GeneID", ""))
            if gene_need and gsym != gene_need:
                continue
            if gene_id_need and gid != gene_id_need:
                continue
            toks = [normalize_variant_token(t) for t in (ent.get("Variant", []) or [])]
            if any(var_need in t for t in toks):  # partial match
                this_ok = True
                break
        if not this_ok:
            return False
    return True

# -----------------------------
# Filtering cells (disease, variants) with fallback
# -----------------------------

def select_cells(df_map: pd.DataFrame, disease_query: str, variant_filters: List[Dict[str, str]]) -> Tuple[List[str], bool]:
    """
    Returns (list_of_cell_names, used_fallback).
    If variant_filters yield no cells, fall back to disease-only.
    """

    def _disease_ok(d: str) -> bool:
        return disease_match_ok(d, disease_query)

    # disease + variants
    cells: List[str] = []
    for _, r in df_map.iterrows():
        if not _disease_ok(str(r.get("Disease", ""))):
            continue
        if variant_filters and not cell_has_all_variants(r, variant_filters):
            continue
        cells.append(str(r.get("CellLine", "")))
    cells = [c.strip() for c in cells if str(c).strip()]
    if cells:
        return sorted(set(cells)), False

    # fallback: disease-only
    cells = []
    for _, r in df_map.iterrows():
        if _disease_ok(str(r.get("Disease",""))):
            c = str(r.get("CellLine","")).strip()
            if c:
                cells.append(c)
    return sorted(set(cells)), True

# -----------------------------
# GDSC / CCLE for LN_IC50
# -----------------------------

def _read_table_auto(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=",", dtype=str, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", dtype=str, engine="python")
        except Exception:
            return pd.DataFrame()

def load_ccle_drugresponse(filepath: str) -> Dict[Tuple[str,str], float]:
    if not filepath or not os.path.exists(filepath):
        return {}
    df_ccle = _read_table_auto(filepath)
    # expected columns: 'gdsc.name','drug','ccle.log.IC50'
    if not {"gdsc.name", "drug", "ccle.log.IC50"}.issubset(df_ccle.columns):
        return {}
    lookup: Dict[Tuple[str,str], float] = {}
    for _, row in df_ccle.iterrows():
        cell_name = str(row.get("gdsc.name", "")).strip()
        drug_name = str(row.get("drug", "")).strip().lower()
        try:
            ln_ic50 = float(row["ccle.log.IC50"])
        except (ValueError, TypeError):
            continue
        if pd.isna(ln_ic50) or cell_name == "" or drug_name == "":
            continue
        lookup[(cell_name, drug_name)] = ln_ic50
    return lookup

def load_gdsc_ic50_multi(filepaths: List[str], ccle_file: str = "") -> Dict[Tuple[str,str], float]:
    lookup: Dict[Tuple[str,str], float] = {}
    for filepath in filepaths:
        if not filepath or not os.path.exists(filepath):
            continue
        df = _read_table_auto(filepath)
        if not {"DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            key = (str(row["CELL_LINE_NAME"]).strip(), str(row["DRUG_NAME"]).strip().lower())
            if key not in lookup:
                try:
                    lookup[key] = float(row["LN_IC50"])
                except (ValueError, TypeError):
                    continue
    # add CCLE if not present
    if ccle_file:
        ccle_lookup = load_ccle_drugresponse(ccle_file)
        for k, v in ccle_lookup.items():
            if k not in lookup:
                lookup[k] = v
    return lookup

# -----------------------------
# GMT parsing
# -----------------------------

def parse_signature(signature: str) -> Tuple[List[str], str, str]:
    """
    Parse a LINCS GMT signature string and return:
      - cell_candidates: possible cell identifiers (token 0 and 1)
      - drug: best-effort extraction
      - direction: "up" / "down" / ""
    Example: CRCGN001_HA1E_24H_A03_geldanamycin_10uM up
    """
    import re as _re

    sig = signature.strip()
    direction = "up" if sig.endswith(" up") else "down" if sig.endswith(" down") else ""
    sig_core = _re.sub(r"\s+(up|down)$", "", sig, flags=_re.IGNORECASE)

    tokens = sig_core.split("_")
    cell_candidates: List[str] = []
    if len(tokens) >= 1:
        cell_candidates.append(tokens[0])   # e.g., CRCGN001
    if len(tokens) >= 2:
        cell_candidates.append(tokens[1])   # e.g., HA1E

    def is_drug_like(tok: str) -> bool:
        t = tok.strip()
        if not _re.search(r"[A-Za-z]", t):
            return False
        if _re.fullmatch(r"\d+\s*(h|hr|hrs|hour|hours|d|day|days|wk|wks)", t, _re.IGNORECASE):
            return False
        if _re.fullmatch(r"[A-H]\d{2}", t, _re.IGNORECASE):
            return False
        if _re.fullmatch(r"\d+(\.\d+)?\s*(pm|nm|um|mm|m|g|kg|mg|ug|ng|pg)", t, _re.IGNORECASE):
            return False
        return True

    drug = "NA"
    for tok in reversed(tokens):
        if is_drug_like(tok):
            drug = tok
            break

    return cell_candidates, drug, direction

# -----------------------------
# Main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Search LINCS GMT by disease+variants; output same schema as previous version."
    )
    ap.add_argument("--input_json", default="", help="JSON with disease_match + variant list (preferred)")
    ap.add_argument("--disease", default="", help="Fallback way to pass disease if not using --input_json")
    ap.add_argument("--mapping_jsonl", default="DB/cell2disease.merge.LINCS.variants.jsonl",
                    help="Cell→Disease/Variant JSONL (one cell per line)")
    ap.add_argument("--extra_jsonl", default="", help="Force-include cells (no variants ok) into final output")
    ap.add_argument("--gmt_folder", required=True, help="Folder containing LINCS .gmt files")
    ap.add_argument("--gdsc_file2", default="", help="GDSC2 fitted dose response CSV/TSV")
    ap.add_argument("--gdsc_file1", default="", help="GDSC1 fitted dose response CSV/TSV")
    ap.add_argument("--ccle_file",  default="", help="CCLE DrugResponse.txt")
    ap.add_argument("--output", required=True, help="Output TSV (same 7 columns as legacy)")
    return ap

def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Read disease + variant filters
    disease_query = ""
    variant_filters: List[Dict[str,str]] = []
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
        disease_query = str(meta.get("disease_match", "")).strip()
        variant_filters = parse_variant_filters(meta.get("variant"))
    else:
        disease_query = args.disease.strip()
        
    if not disease_query:
        raise ValueError("Please provide disease via --input_json (disease_match) or --disease")

    # Mapping + extras
    df_map = load_cell2disease_jsonl(args.mapping_jsonl)
    df_extra = load_extra_jsonl(args.extra_jsonl) if args.extra_jsonl else pd.DataFrame(columns=["CellLine","Disease","Variant"])

    # Select cells (with fallback)
    cells_selected, used_fallback = select_cells(df_map, disease_query, variant_filters)

    # Add forced extras
    for c in df_extra["CellLine"].dropna().astype(str).tolist():
        c = c.strip()
        if c and c not in cells_selected:
            cells_selected.append(c)

    if not cells_selected:
        print("[WARN] No cells matched disease/variants and no extras provided. Output may be empty.")

    # Build normalized set for GMT matching (legacy-style substring check)
    norm_targets = set(norm_cell(c) for c in cells_selected if str(c).strip())

    # Build drug response lookup
    gdsc_lookup = load_gdsc_ic50_multi([args.gdsc_file2, args.gdsc_file1], ccle_file=args.ccle_file)

    # Scan GMT and collect results for matched cells
    results: List[Dict[str, Any]] = []
    gmt_files = glob.glob(os.path.join(args.gmt_folder, "*.gmt"))
    for gmt_file in gmt_files:
        with open(gmt_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                signature, _, *genes = parts
                cell_candidates, drug, direction = parse_signature(signature)
                cand_norms = [norm_cell(c) for c in cell_candidates if c]

                # match if any candidate cell equals or is substring-overlapping with targets
                def match_any(cands: List[str], targets: Set[str]) -> bool:
                    for cn in cands:
                        for tn in targets:
                            if cn == tn or (cn in tn) or (tn in cn):
                                return True
                    return False

                if match_any(cand_norms, norm_targets):
                    # choose the closest-looking cell name for output (prefer earlier tokens)
                    cell_for_output = cell_candidates[0]
                    for c in cell_candidates:
                        nc = norm_cell(c)
                        if any(nc == t or nc in t or t in nc for t in norm_targets):
                            cell_for_output = c
                            break

                    ln_ic50 = gdsc_lookup.get((cell_for_output, str(drug).lower()), "")
                    ic50 = ln_to_ic50(ln_ic50) if ln_ic50 != "" else ""
                    results.append({
                        "Signature": signature,
                        "LN_IC50": ln_ic50,
                        "IC50": ic50,
                        "CellLine": cell_for_output,
                        "Drug": drug,
                        "Direction": direction,
                        "Genes": ",".join(genes)
                    })

    # Append extras that had no GMT rows so they still appear in final TSV
    # (blank GMT-derived fields; keeps the same 7 columns)
    have_norms = set(norm_cell(r["CellLine"]) for r in results)
    for _, row in df_extra.iterrows():
        c = str(row.get("CellLine","")).strip()
        if not c:
            continue
        if norm_cell(c) in have_norms:
            continue  # already present via GMT
        results.append({
            "Signature": "",
            "LN_IC50": "",
            "IC50": "",
            "CellLine": c,
            "Drug": "",
            "Direction": "",
            "Genes": ""
        })

    # If still nothing matched (e.g., no GMT signatures for selected cells), we can also
    # emit disease-only cells as blank rows to avoid an empty file.
    if not results and cells_selected:
        for c in sorted(cells_selected, key=lambda x: x.lower()):
            results.append({
                "Signature": "",
                "LN_IC50": "",
                "IC50": "",
                "CellLine": c,
                "Drug": "",
                "Direction": "",
                "Genes": ""
            })

    # Write TSV (same schema)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=["Signature","LN_IC50","IC50","CellLine","Drug","Direction","Genes"])\
      .to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(results)} records to {out_path}")
    if used_fallback:
        print("[INFO] Used disease-only fallback (no cells matched variant filters).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



"""
python 4.1.LINCS.search.py \
  --input_json input/coloncancer.json \
  --mapping_jsonl DB/cell2disease.merge.LINCS.variants.jsonl \
  --gmt_folder DB/LINCS/gmt \
  --gdsc_file2 DB/GDSC/GDSC2_fitted_dose_response_27Oct23.csv \
  --gdsc_file1 DB/GDSC/GDSC1_fitted_dose_response_27Oct23.csv \
  --ccle_file DB/CCLE/DrugResponse.txt \
  --output output/LINCS/coloncancer/LINCS.search.coloncancer.cell2gene.tsv

python 4.1.LINCS.search.py \
  --input_json input/melanoma.json \
  --mapping_jsonl DB/cell2disease.merge.LINCS.variants.jsonl \
  --gmt_folder DB/LINCS/gmt \
  --gdsc_file2 DB/GDSC/GDSC2_fitted_dose_response_27Oct23.csv \
  --gdsc_file1 DB/GDSC/GDSC1_fitted_dose_response_27Oct23.csv \
  --ccle_file DB/CCLE/DrugResponse.txt \
  --output output/LINCS/melanoma/LINCS.search.melanoma.cell2gene.tsv
  
python 4.1.LINCS.search.py \
  --input_json input/breastcancer.json \
  --mapping_jsonl DB/cell2disease.merge.LINCS.variants.jsonl \
  --gmt_folder DB/LINCS/gmt \
  --gdsc_file2 DB/GDSC/GDSC2_fitted_dose_response_27Oct23.csv \
  --gdsc_file1 DB/GDSC/GDSC1_fitted_dose_response_27Oct23.csv \
  --ccle_file  DB/CCLE/DrugResponse.txt \
  --output output/LINCS/breastcancer/LINCS.search.breastcancer.cell2gene.tsv

"""
