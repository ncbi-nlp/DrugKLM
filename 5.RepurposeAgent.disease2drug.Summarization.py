#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch payload builder for disease-drug using:
  1) disease2drug candidates (JSONL) <-- main list
  2) optional drug2gene evidence (JSONL; scanned by folder, matched by "Drug" field)
  3) FDA labels.tsv and drugs.csv
  4) optional mechanistic JSON files (per drug; folder of JSON)

Outputs one JSON payload file per (disease, drug), WITHOUT calling GPT.
Automatically prunes empty fields (empty dict/list/"").
"""

import os, json, argparse, csv, re
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------
# IO helpers
# ---------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip invalid JSON at {path} line {i}: {e}")
    return items

# ---------------------------
# FDA helpers
# ---------------------------

def normalize_drug_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[-,.;:/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_fda_drugs_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows

def match_fda_drugs_by_brand_sponsor(rows: List[Dict[str, str]],
                                     drug_name: str,
                                     sponsor_name: Optional[str] = None) -> List[Dict[str, str]]:
    dn = normalize_drug_name(drug_name)
    if not dn:
        return []
    sponsor_lc = (sponsor_name or "").strip().lower()
    hits: List[Dict[str, str]] = []
    for r in rows:
        brand_lc = normalize_drug_name(r.get("brand_name", ""))
        if dn != brand_lc:
            continue
        if sponsor_lc and sponsor_lc not in (r.get("sponsor_name", "") or "").lower():
            continue
        hits.append(r)
    return hits

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def load_fda_labels_tsv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            clean = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            rows.append(clean)
    return rows

def match_fda_entries_for_drug(labels: List[Dict[str, str]], drug_name: str) -> List[Dict[str, str]]:
    dn = _norm(drug_name)
    hits: List[Dict[str, str]] = []
    for r in labels:
        b = _norm(r.get("brand_name", ""))
        g = _norm(r.get("generic_name", ""))
        if dn and (dn == b or dn == g):
            hits.append(r)
    return hits

def dedup_join(texts: List[str]) -> str:
    uniq: List[str] = []
    seen = set()
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return "\n\n---\n\n".join(uniq)

def add_fda_to_d2d(d2d_obj: Dict[str, Any],
                   fda_hits: List[Dict[str, str]],
                   disease_name: str) -> Dict[str, Any]:
    out = dict(d2_obj)
    if not fda_hits:
        out["FDA-approved"] = False
        out[f"FDA-approved to {disease_name}"] = False
        return out
    inds = [r.get("indications_and_usage", "") for r in fda_hits]
    contras = [r.get("contraindications", "") for r in fda_hits]
    warns = [r.get("warnings", "") for r in fda_hits]
    out["FDA_indications_and_usage"] = dedup_join(inds)
    out["FDA_contraindications"] = dedup_join(contras)
    out["FDA_warnings"] = dedup_join(warns)
    out["FDA-approved"] = True
    disease_lc = disease_name.strip().lower()
    ind_blob_lc = out.get("FDA_indications_and_usage", "").lower()
    out[f"FDA-approved to {disease_name}"] = (disease_lc in ind_blob_lc) if disease_lc else False
    return out

def derive_marketing_flags(hits: List[Dict[str, str]]) -> Dict[str, Any]:
    if not hits:
        return {
            "FDA-approved": False,
            "FDA_currently_marketed": False,
            "FDA_sponsors": [],
            "FDA_marketing_statuses": [],
            "FDA_brand_names": []
        }
    sponsors = sorted({(h.get("sponsor_name") or "").strip() for h in hits if (h.get("sponsor_name") or "").strip()})
    statuses = sorted({(h.get("marketing_status") or "").strip() for h in hits if (h.get("marketing_status") or "").strip()})
    brands = sorted({(h.get("brand_name") or "").strip() for h in hits if (h.get("brand_name") or "").strip()})
    currently_marketed = any((s or "").lower() != "discontinued" for s in statuses if s)
    return {
        "FDA-approved": True,
        "FDA_currently_marketed": bool(currently_marketed),
        "FDA_sponsors": sponsors,
        "FDA_marketing_statuses": statuses,
        "FDA_brand_names": brands
    }

# ---------------------------
# Gene evidence summary
# ---------------------------

def build_gene_evidence_summary(drug_gene_list: List[Dict[str, Any]],
                                max_genes: int,
                                max_chars: int,
                                drop_empty: bool = True) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for obj in drug_gene_list:
        gene = str(obj.get("Gene", obj.get("gene", ""))).strip()
        if not gene:
            continue
        summ = str(obj.get("evidence_summary", "") or "").strip()
        mech = str(obj.get("mechanism", "") or "").strip()
        if not summ and drop_empty and not mech:
            continue
        if isinstance(summ, str) and max_chars > 0 and len(summ) > max_chars:
            summ = summ[:max_chars] + " ..."
        text = f"{mech} | {summ}" if mech and summ else (mech or summ)
        out[gene] = text
        if max_genes > 0 and len(out) >= max_genes:
            break
    return out

# ---------------------------
# Mechanistic JSON loader
# ---------------------------

def load_gsea_mechanistic_dir(folder: Optional[str], disease_filter: Optional[str] = None):
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not folder or not os.path.isdir(folder):
        return grouped
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] skip invalid JSON at {path}: {e}")
            continue
        drug = str(obj.get("Drug", "")).strip()
        disease = str(obj.get("Disease", "")).strip()
        if not drug or not disease:
            continue
        if disease_filter and disease.lower() != disease_filter.lower():
            continue
        key = (disease.lower(), drug.lower())
        grouped[key] = {
            "disease": disease,
            "drug": drug,
            "mechanistic_verdict": obj.get("mechanistic_verdict", {}),
            "mechanistic_statement": obj.get("mechanistic_statement", ""),
            "selected_up_pathways": obj.get("selected_up_pathways", []),
            "selected_down_pathways": obj.get("selected_down_pathways", [])
        }
    return grouped

# ---------------------------
# Prune helper
# ---------------------------

def _prune_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        pruned = {}
        for k, v in obj.items():
            v2 = _prune_empty(v)
            if v2 not in ([], {}, ""):
                pruned[k] = v2
        return pruned
    if isinstance(obj, list):
        lst = [_prune_empty(v) for v in obj]
        lst = [v for v in lst if v not in ([], {}, "")]
        return lst
    return obj

# ---------------------------
# Payload builder
# ---------------------------

def build_input_payload_selected_paths(disease_drug_obj: Dict[str, Any],
                                       gene_evidence_summary: Dict[str, str],
                                       disease: str,
                                       drug: str,
                                       gsea_mechanistic: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "Disease": disease,
        "Drug": drug,
        "disease_drug_evidence": disease_drug_obj,
        "gene_evidence_summary": gene_evidence_summary,
        "gsea_mechanistic": gsea_mechanistic
    }
    return _prune_empty(payload)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease2drug_jsonl", required=True)
    ap.add_argument("--drug2gene_dir", default=None)
    ap.add_argument("--gsea_dir", default=None, help="Folder containing mechanistic JSON files (per drug)")
    ap.add_argument("--disease", default=None)
    ap.add_argument("--fda_labels_tsv", default="DB/FDA-approved/labels.tsv")
    ap.add_argument("--fda_drugs_csv", default="DB/FDA-approved/drugs.csv")
    ap.add_argument("--max_genes", type=int, default=100)
    ap.add_argument("--max_chars", type=int, default=200)
    ap.add_argument("--keep_empty_summaries", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    d2d_records = read_jsonl(args.disease2drug_jsonl)

    try:
        fda_labels = load_fda_labels_tsv(args.fda_labels_tsv)
    except Exception as e:
        print(f"[WARN] FDA labels not loaded: {e}")
        fda_labels = []
    try:
        fda_drugs_rows = load_fda_drugs_csv(args.fda_drugs_csv)
    except Exception as e:
        print(f"[WARN] FDA drugs.csv not loaded: {e}")
        fda_drugs_rows = []

    gsea_grouped = load_gsea_mechanistic_dir(args.gsea_dir, args.disease)

    os.makedirs(args.out_dir, exist_ok=True)

    def enrich_fda(d2d_obj: Dict[str, Any], disease: str, drug: str) -> Dict[str, Any]:
        try:
            hits_by_brand = match_fda_drugs_by_brand_sponsor(fda_drugs_rows, drug, sponsor_name=None)
            flags = derive_marketing_flags(hits_by_brand)
        except Exception:
            flags = derive_marketing_flags([])
        enriched = dict(d2d_obj)
        for k, v in flags.items():
            enriched[k] = v
        try:
            hits = match_fda_entries_for_drug(fda_labels, drug)
            tmp = add_fda_to_d2d(enriched, hits, disease)
            tmp["FDA-approved"] = enriched.get("FDA-approved", False)
            tmp["FDA_currently_marketed"] = enriched.get("FDA_currently_marketed", False)
            tmp["FDA_sponsors"] = enriched.get("FDA_sponsors", [])
            tmp["FDA_marketing_statuses"] = enriched.get("FDA_marketing_statuses", [])
            tmp["FDA_brand_names"] = enriched.get("FDA_brand_names", [])
            enriched = tmp
        except Exception:
            enriched.setdefault(f"FDA-approved to {disease}", False)
        for _k in ("FDA_currently_marketed", "FDA_sponsors", "FDA_marketing_statuses", "FDA_brand_names"):
            enriched.pop(_k, None)
        return enriched

    def try_load_drug2gene(drug: str) -> List[Dict[str, Any]]:
        if not args.drug2gene_dir:
            return []
        drug_lc = drug.lower()
        collected: List[Dict[str, Any]] = []
        for fname in os.listdir(args.drug2gene_dir):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(args.drug2gene_dir, fname)
            try:
                rows = read_jsonl(path)
                for r in rows:
                    if str(r.get("Drug", "")).lower() == drug_lc:
                        collected.append(r)
            except Exception as e:
                print(f"[WARN] failed to read {path}: {e}")
        return collected

    for d2d_obj in d2d_records:
        disease = str(d2d_obj.get("Disease", "")).strip()
        drug = str(d2d_obj.get("Drug", "")).strip()

        # Skip entries with missing or placeholder drug names
        if not disease or not drug:
            continue
        if drug.upper() in {"N/A", "NA"}:
            print(f"[INFO] Skip placeholder drug name: {drug} for disease {disease}")
            continue

        if args.disease and disease.lower() != args.disease.lower():
            continue

        d2d_obj_enriched = enrich_fda(d2d_obj, disease, drug)

        d2g_list = try_load_drug2gene(drug)
        print(f"[DEBUG] Loaded {len(d2g_list)} gene entries for {drug}")
        gene_summary_map = build_gene_evidence_summary(
            d2g_list,
            max_genes=args.max_genes,
            max_chars=args.max_chars,
            drop_empty=not args.keep_empty_summaries,
        )

        gsea_bundle = gsea_grouped.get((disease.lower(), drug.lower()), {})

        payload = build_input_payload_selected_paths(
            disease_drug_obj=d2d_obj_enriched,
            gene_evidence_summary=gene_summary_map,
            disease=disease,
            drug=drug,
            gsea_mechanistic=gsea_bundle,
        )

        # Build safe output filename (remove characters illegal for filesystem)
        safe_disease = _norm(disease)
        safe_drug = _norm(drug)  # critical fix: prevents "/" from breaking directories

        out_subdir = os.path.join(args.out_dir)
        os.makedirs(out_subdir, exist_ok=True)

        # Now create a safe filename
        out_filename = f"{safe_disease}.{safe_drug}.input.json"
        out_path = os.path.join(out_subdir, out_filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved payload: {out_path}")
        
if __name__ == "__main__":
    main()



"""
python 5.KGxLM.disease2drug.Summarization.py \
  --disease2drug_jsonl output/disease2drug/breastcancer.disease2drug.candidate.jsonl \
  --drug2gene_dir output/drug2gene/breastcancer \
  --gsea_dir output/LINCS_enrich_summary_mechanistic/breastcancer \
  --out_dir output/final_input/breastcancer

python 5.KGxLM.disease2drug.Summarization.py \
  --disease2drug_jsonl output/disease2drug/glioblastoma.disease2drug.candidate.jsonl \
  --drug2gene_dir output/drug2gene/glioblastoma \
  --gsea_dir output/LINCS_enrich_summary_mechanistic/glioblastoma \
  --out_dir output/final_input/glioblastoma

python 5.KGxLM.disease2drug.Summarization.py \
  --disease2drug_jsonl output/disease2drug/mcrc.disease2drug.candidate.jsonl \
  --drug2gene_dir output/drug2gene/mcrc \
  --gsea_dir output/LINCS_enrich_summary_mechanistic/mcrc \
  --out_dir output/final_input/mcrc
  
python 5.KGxLM.disease2drug.Summarization.py \
  --disease2drug_jsonl output/disease2drug/melanoma.disease2drug.candidate.jsonl \
  --drug2gene_dir output/drug2gene/melanoma \
  --gsea_dir output/LINCS_enrich_summary_mechanistic/melanoma \
  --out_dir output/final_input/melanoma

"""
