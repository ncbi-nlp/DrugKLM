#!/usr/bin/env python3

import re
import os
import argparse
import csv
import sys
import json
import requests
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

# ----------------------
# Hard-coded file paths
# ----------------------
INTERVENTION_OTHER_NAMES = "DB/ClinicalTrials/intervention_other_names.txt"
INTERVENTIONS = "DB/ClinicalTrials/interventions.txt"
BROWSE_CONDITIONS = "DB/ClinicalTrials/browse_conditions.txt"
NCT_META_FILE = "DB/ClinicalTrials/nct_phase_studytype.txt"
FDA_LABELS_TSV = "DB/FDA-approved/labels.tsv"

PUBMED_SESSION = requests.Session()
PUBMED_SLEEP_SEC = 0.4
PUBMED_MAX_RETRY = 3

CATEGORY_NAME_MAP = {
    1:  "FDA-Approved for [Disease]",
    2:  "Positive results in Phase III (for [Disease])",
    3:  "In Phase III (for [Disease])",
    4:  "Failed in Phase III (for [Disease])",
    5:  "Positive results in Phase II (for [Disease])",
    6:  "In Phase II (for [Disease])",
    7:  "Failed in Phase II (for [Disease])",
    8:  "Positive results in Phase I (for [Disease])",
    9:  "In Phase I (for [Disease])",
    10: "Failed in Phase I (for [Disease])",
    11: "Positive result in animal study (for [Disease])",
    12: "Negative result in animal study (for [Disease])",
    13: "Positive in vivo result (for [Disease])",
    14: "Negative in vivo result (for [Disease])",
    15: "Positive in vitro result (for [Disease])",
    16: "Negative in vitro result (for [Disease])",
    17: "Rarely discussed / insufficient evidence (for [Disease])"
}

DISEASE_SYNONYM_CACHE = {}

# ======================
# Azure OpenAI helpers
# ======================
def generate_disease_synonyms(config, deployment_name, disease_name):
    """
    Generate disease synonyms using GPT-4o (live version).
    """

    prompt = f"""
You are a biomedical terminology assistant.

Given the disease:

{disease_name}

Generate clinically relevant disease names including:

1. Exact disease synonyms
2. Widely used clinical abbreviations
3. Major research-relevant subtypes that are:
   - Commonly used in ClinicalTrials.gov
   - Frequently used in PubMed article titles

Rules:
- Include only disease entity names used in clinical research.
- Exclude rare pathological descriptions.
- Exclude long descriptive phrases ending with "of the pancreas".
- Exclude highly specific histologic variants rarely used in trials.
- Do NOT include biomarkers, mutations, or drug names.
- Keep entries concise.

Return ONLY valid JSON:

{{
  "disease_list": ["name1", "name2", ...]
}}
"""

    raw_response = call_azure_gpt4o(prompt, config)

    parsed = safe_parse_json(raw_response)

    disease_list = []

    if isinstance(parsed, dict) and "disease_list" in parsed:
        disease_list = parsed["disease_list"]

    if not disease_list:
        return [disease_name.lower()]

    if disease_name not in disease_list:
        disease_list.append(disease_name)

    return sorted(set(x.lower().strip() for x in disease_list if x))



def load_azure_openai_config(param_file="parameter.gpt4o.real.txt"):
    """
    Expected keys (tab or = separated):
      AZURE_OPENAI_ENDPOINT
      API_KEY
      DEPLOYMENT_NAME
      API_VERSION
    """
    config = {}
    with open(param_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                k, v = line.split("\t", 1)
            elif "=" in line:
                k, v = line.split("=", 1)
            else:
                continue
            config[k.strip()] = v.strip()

    required = ["AZURE_OPENAI_ENDPOINT", "API_KEY", "DEPLOYMENT_NAME", "API_VERSION"]
    for k in required:
        if k not in config:
            raise RuntimeError(f"Missing {k} in parameter.gpt4o.txt")

    return config


def call_azure_gpt4o(prompt_text, config):
    endpoint = config["AZURE_OPENAI_ENDPOINT"].rstrip("/")

    url = (
        f"{endpoint}/openai/deployments/"
        f"{config['DEPLOYMENT_NAME']}/chat/completions"
        f"?api-version={config['API_VERSION']}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": config["API_KEY"],
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a careful biomedical evidence classifier."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0,
        "max_completion_tokens": 2048
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def safe_parse_json(text):
    """
    Robust JSON parser for GPT output.
    Accepts:
      - pure JSON
      - ```json ... ```
      - text + JSON
    """
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

    return {
        "Error": "Failed to parse GPT JSON",
        "RawResponse": text[:1000]
    }


# ======================
# ClinicalTrials.gov helpers
# ======================
def load_nct_metadata():
    meta = {}

    with open(NCT_META_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            meta[row["nct_id"]] = {
                "study_type": row["study_type"],
                "phase": row["phase"],
                "overall_status": row["overall_status"],
                "last_update_posted": row["last_update_posted_date"],
            }

    return meta
    
def get_study_json(nct_id, cache):
    if nct_id in cache:
        return cache[nct_id]

    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    study = r.json()
    cache[nct_id] = study
    return study


def extract_phase_status(study):
    phases = (
        study.get("protocolSection", {})
        .get("designModule", {})
        .get("phases", [])
    )

    overall_status = (
        study.get("protocolSection", {})
        .get("statusModule", {})
        .get("overallStatus", "")
    )

    has_results = study.get("hasResults", False)

    last_update_posted = (
        study.get("protocolSection", {})
        .get("statusModule", {})
        .get("lastUpdatePostDateStruct", {})
        .get("date", "")
    )

    return "|".join(phases), overall_status, has_results, last_update_posted


def categorize_trial(overall_status, has_results, pmid_found):
    if overall_status != "COMPLETED":
        return "Ongoing"
    if has_results or pmid_found:
        return "Completed - results published"
    return "Completed - results not yet available"


# ======================
# PubMed helpers
# ======================
def search_pubmed_by_drug_disease_union(drug, disease_list, top_n=5):
    """
    Try full union query first (no field restriction).
    If fails or returns no results, fallback to top 20 terms.
    """

    if not disease_list:
        return None

    # Clean disease terms
    clean_terms = []

    for d in disease_list:
        if len(d) <= 3:
            continue

        # remove all parentheses and their content
        d = re.sub(r"\(.*?\)", "", d)

        # remove quotes
        d = d.replace('"', '')

        # normalize whitespace
        d = " ".join(d.split()).strip()

        if d:
            clean_terms.append(d)

    if not clean_terms:
        return None

    safe_drug = drug.replace('"', '').strip()

    # ----------------------------
    # FULL QUERY (all terms)
    # ----------------------------
    disease_query_full = " OR ".join(
        f'"{d}"'
        for d in clean_terms
    )

    full_query = f'"{safe_drug}" AND ({disease_query_full})'

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": full_query,
        "sort": "pub+date",
        "retmax": top_n,
        "retmode": "json",
        "email": EMAIL,
        "tool": TOOL
    }

    try:
        # Use POST to avoid URL length issues
        r = PUBMED_SESSION.post(search_url, data=params)
        r.raise_for_status()
        data = r.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        print(pmids)
        
        if pmids:
            return pmids[0]

    except Exception:
        pass

    # ----------------------------
    # FALLBACK TOP 20
    # ----------------------------
    limited_terms = clean_terms[:20]

    disease_query_limited = " OR ".join(
        f'"{d}"'
        for d in limited_terms
    )

    fallback_query = f'"{safe_drug}" AND ({disease_query_limited})'

    params["term"] = fallback_query

    try:
        r = PUBMED_SESSION.post(search_url, data=params)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [None])[0]
    except Exception:
        return None


def search_pubmed_by_drug_disease(drug, disease, top_n=5):
    """
    Search PubMed using drug + disease keywords,
    return the most recent PMID.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{drug}[Title/Abstract] AND {disease}[Title/Abstract]",
        "sort": "pub+date",
        "retmax": top_n,
        "retmode": "json",
        "tool": "pubmed_recent_fetch"
    }

    try:
        r = pubmed_get(search_url, params)
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [None])[0]
    except Exception:
        return None

def fetch_pubmed_record(pmid):
    """
    Return PubMed record as structured JSON.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "tool": "pubmed_recent_fetch"
    }

    try:
        r = pubmed_get(url, params)
        root = ET.fromstring(r.text)
    except Exception:
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        return None

    title = article.findtext(".//ArticleTitle") or ""
    abstract = " ".join(
        p.text for p in article.findall(".//AbstractText") if p.text
    )

    return {
        "pmid": pmid,
        "title": title.strip(),
        "abstract": abstract.strip()
    }
    
def fetch_pubmed_records(pmids):
    """
    Fetch multiple PubMed records.

    Accepts either:
      - a list of PMIDs (list[str])
      - a single PMID string (str)
    """
    if not pmids:
        return []

    # If a single PMID string is provided, wrap it into a list
    if isinstance(pmids, str):
        pmids = [pmids]

    records = []
    for pmid in pmids:
        rec = fetch_pubmed_record(pmid)
        if rec:
            records.append(rec)
    return records
    
def pubmed_get(url, params):
    for attempt in range(PUBMED_MAX_RETRY):
        r = PUBMED_SESSION.get(url, params=params)
        if r.status_code == 200:
            time.sleep(PUBMED_SLEEP_SEC)
            return r
        if r.status_code == 429:
            time.sleep((attempt + 1) * 2)
            continue
        r.raise_for_status()
    raise RuntimeError("PubMed API failed after retries")


def search_pubmed_by_nct(nct_id, top_n=5):
    """
    Search PubMed by NCT ID and return a list of up to top_n PMIDs.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": nct_id,
        "retmax": top_n,
        "retmode": "json",
        "tool": "pubmed_recent_fetch"
    }

    try:
        r = pubmed_get(search_url, params)
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def fetch_title_abstract_by_pmid(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "tool": "pubmed_recent_fetch"
    }

    try:
        r = pubmed_get(url, params)
        root = ET.fromstring(r.text)
    except Exception:
        return ""

    article = root.find(".//PubmedArticle")
    if article is None:
        return ""

    title = article.findtext(".//ArticleTitle") or ""
    abstract = " ".join(
        p.text for p in article.findall(".//AbstractText") if p.text
    )

    return f"{title} || {abstract}".strip()


# ======================
# Local file loaders
# ======================
def load_all_drugs():
    drugs_by_nct = defaultdict(set)

    with open(INTERVENTION_OTHER_NAMES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            drugs_by_nct[row["nct_id"]].add(row["name"])

    with open(INTERVENTIONS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            drugs_by_nct[row["nct_id"]].add(row["name"])

    return drugs_by_nct


def load_all_diseases():
    diseases_by_nct = defaultdict(set)

    with open(BROWSE_CONDITIONS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            diseases_by_nct[row["nct_id"]].add(row["mesh_term"])

    return diseases_by_nct


# ======================
# FDA-approved labels helpers
# ======================
def _norm_text(s: str) -> str:
    """Normalize text for case-insensitive exact match. Keep it simple and deterministic."""
    if s is None:
        return ""
    return str(s).strip().lower()


def load_fda_labels(labels_tsv: str = FDA_LABELS_TSV):
    """
    Load FDA labels TSV and build an index for case-insensitive exact match against:
      - brand_name
      - generic_name
    Returns:
      - name2rows: dict[str, list[dict]]
    """
    name2rows = defaultdict(list)
    try:
        with open(labels_tsv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                brand = _norm_text(row.get("brand_name", ""))
                generic = _norm_text(row.get("generic_name", ""))
                if brand:
                    name2rows[brand].append(row)
                if generic:
                    name2rows[generic].append(row)
    except FileNotFoundError:
        # If the FDA label file is not available, fall back to GPT behavior silently.
        return defaultdict(list)
    return name2rows


def fda_label_match(drug: str, disease_list: list, name2rows) -> dict:

    drug_key = _norm_text(drug)
    rows = name2rows.get(drug_key, [])

    if not rows:
        return {
            "drug_matched": False,
            "disease_in_indication": False,
            "matched_rows": 0,
            "indication_text": ""
        }

    for r in rows:
        indication_norm = _norm_text(r.get("indications_and_usage", ""))

        for synonym in disease_list:
            if synonym in indication_norm:
                return {
                    "drug_matched": True,
                    "disease_in_indication": True,
                    "matched_rows": len(rows),
                    "indication_text": r.get("indications_and_usage", "")
                }

    return {
        "drug_matched": True,
        "disease_in_indication": False,
        "matched_rows": len(rows),
        "indication_text": rows[0].get("indications_and_usage", "")
    }

def process_drug_disease_pair(
    drug,
    disease_list,
    drugs_by_nct,
    diseases_by_nct,
    nct_meta
):
    disease_list = [x.lower() for x in disease_list]
    drug_l = drug.lower()

    matched_ncts = sorted(
        nct for nct in drugs_by_nct
        if any(drug_l in d.lower() for d in drugs_by_nct[nct])
        and any(
            any(synonym in mesh_l for synonym in disease_list)
            for mesh_l in (m.lower() for m in diseases_by_nct.get(nct, []))
        )
    )

    # If too many NCTs, keep only those with valid phase
    if len(matched_ncts) > 50:

        phase_rank = {
            "PHASE4": 4,
            "PHASE3": 3,
            "PHASE2": 2,
            "PHASE1": 1,
            "EARLYPHASE1": 0
        }

        ranked_trials = []

        for nct_id in matched_ncts:
            meta = nct_meta.get(nct_id, {})
            phase = meta.get("phase", "")

            if not phase:
                continue

            phase = phase.upper().replace(" ", "")

            if phase == "NA":
                continue

            # Handle combined phases like PHASE1/PHASE2
            phase_parts = phase.split("/")

            max_rank = max(
                phase_rank.get(p, -1)
                for p in phase_parts
            )

            if max_rank >= 0:
                try:
                    nct_number = int(nct_id.replace("NCT", ""))
                except Exception:
                    nct_number = 0

                ranked_trials.append((max_rank, nct_number, nct_id))

        # 排序：
        # 1️⃣ Phase rank 由高到低
        # 2️⃣ NCT 數字由大到小（較新的 trial 優先）
        ranked_trials.sort(key=lambda x: (-x[0], -x[1]))

        # 取前 50
        if ranked_trials:
            matched_ncts = [x[2] for x in ranked_trials[:50]]
        else:
            matched_ncts = matched_ncts[:50]

    print(matched_ncts)

    trials = []

    for nct_id in matched_ncts:
        meta = nct_meta.get(nct_id, {})

        phases = meta.get("phase", "")
        overall_status = meta.get("overall_status", "")
        last_update_posted = meta.get("last_update_posted", "")

        pmids = search_pubmed_by_nct(nct_id, top_n=5)
        pubmed_records = fetch_pubmed_records(pmids) if pmids else []

        record = {
            "nct_id": nct_id,
            "drugs": "|".join(sorted(drugs_by_nct[nct_id])),
            "mono_combo": "mono" if len(drugs_by_nct[nct_id]) == 1 else "combo",
            "diseases": "|".join(sorted(diseases_by_nct.get(nct_id, []))),
            "last_update_posted": last_update_posted,
            "phases": phases,
            "results_category": categorize_trial(
                overall_status,
                has_results=False,
                pmid_found=bool(pmids)
            )
        }

        if pubmed_records:
            record["pmids"] = pmids
            record["pubmed"] = pubmed_records

        trials.append(record)

    return trials

def load_prompt_template(prompt_file):
    with open(prompt_file, encoding="utf-8") as f:
        return f.read()


def build_prompt(prompt_template, drug, disease, evidence_json):
    return (
        prompt_template
        .replace("[Drug]", drug)
        .replace("[Disease]", disease)
        .replace("[Input]", json.dumps(evidence_json, ensure_ascii=False, indent=2))
    )


# ======================
# Main
# ======================
def main(input_tsv, output_gpt_jsonl, prompt_file):

    nct_meta = load_nct_metadata()
    
    CATEGORIZATION_FIELDS = [
        "FDA-Approved",
        "FDA-Approved Indication",
        "Mono/Combo Therapy",
        "Category #",
        "Category Name",
        "Reference"
    ]

    if os.path.exists("disease_synonym_cache.json"):
        with open("disease_synonym_cache.json", encoding="utf-8") as f:
            DISEASE_SYNONYM_CACHE.update(json.load(f))
    for k in list(DISEASE_SYNONYM_CACHE.keys()):
        DISEASE_SYNONYM_CACHE[k] = [
            x.lower() for x in DISEASE_SYNONYM_CACHE[k]
        ]
    
    print("Loading drugs / diseases...")
    drugs_by_nct = load_all_drugs()
    diseases_by_nct = load_all_diseases()

    print("Loading FDA labels...")
    fda_name2rows = load_fda_labels(FDA_LABELS_TSV)

    print("Loading Azure config and prompt...")
    azure_config = load_azure_openai_config()
    prompt_template = load_prompt_template(prompt_file)

    with open(input_tsv, newline="", encoding="utf-8") as fin, \
         open(output_gpt_jsonl, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + CATEGORIZATION_FIELDS
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, row in enumerate(reader, 1):
            disease = row["disease"].strip()
            drug = row["drug"].strip()

            print(f"[{i}] {disease} | {drug}")

            # =============================
            # Step 0: Disease synonym expansion (cached)
            # =============================
            if disease not in DISEASE_SYNONYM_CACHE:
                disease_list = generate_disease_synonyms(
                    azure_config,
                    azure_config["DEPLOYMENT_NAME"],
                    disease
                )
                DISEASE_SYNONYM_CACHE[disease] = disease_list
            else:
                disease_list = DISEASE_SYNONYM_CACHE[disease]
                
            # =============================
            # Step 1: Collect trial evidence
            # =============================
            trials = process_drug_disease_pair(
                    drug,
                    disease_list,
                    drugs_by_nct,
                    diseases_by_nct,
                    nct_meta
                )

            if trials:
                evidence_obj = {
                    "input_disease": disease,
                    "input_drug": drug,
                    "trials": trials
                }
            else:
                pmids = search_pubmed_by_drug_disease_union(
                    drug,
                    disease_list,
                    top_n=5
                )

                pubmed_records = fetch_pubmed_records(pmids) if pmids else []

                evidence_obj = {
                    "input_disease": disease,
                    "input_drug": drug,
                    "pubmed": pubmed_records
                }

            # =============================
            # Step 2: FDA label rule (before GPT)
            # =============================
            label_hit = fda_label_match(drug, disease_list, fda_name2rows)

            if label_hit["drug_matched"]:
                evidence_obj["fda_label_info"] = {
                    "is_fda_approved": True,
                    "indications_and_usage": label_hit.get("indication_text", "")
                }

            # =============================
            # Step 3: Build prompt AFTER evidence complete
            # =============================
            prompt_text = build_prompt(
                prompt_template, drug, disease, evidence_obj
            )

            try:
                gpt_response = call_azure_gpt4o(prompt_text, azure_config)
                gpt_json = safe_parse_json(gpt_response)
            except Exception:
                gpt_json = {}

            status_cat = gpt_json.get("Status Category", "")

            # =============================
            # Step 4: FDA override logic
            # =============================
            fda_approved = gpt_json.get("FDA-Approved", "")
            fda_approved_indication = gpt_json.get("FDA_Approved_Indication", "")

            # A) Deterministic FDA label rule
            if label_hit["drug_matched"]:
                fda_approved = "Yes"

                if label_hit["disease_in_indication"]:
                    fda_approved_indication = disease

            # B) Existing override rules
            input_fda_status = row.get("FDA_status", "").lower()

            if input_fda_status.startswith("fda-approved"):
                fda_approved = "Yes"
            else:
                try:
                    if int(status_cat) == 1:
                        fda_approved = "Yes"
                except Exception:
                    pass

            # =============================
            # Step 5: Write output
            # =============================
            row["FDA-Approved"] = fda_approved
            row["FDA-Approved Indication"] = fda_approved_indication
            row["Mono/Combo Therapy"] = gpt_json.get("Mono/Combo Therapy", "")
            row["Category #"] = status_cat
            row["Reference"] = gpt_json.get("Reference", "")

            # Derive Category Name
            category_name = ""
            try:
                status_cat_int = int(status_cat)
                template = CATEGORY_NAME_MAP.get(status_cat_int, "")
                if template:
                    category_name = template.replace("[Disease]", disease)
            except Exception:
                pass

            row["Category Name"] = category_name

            writer.writerow(row)
            fout.flush()

    # Clean all ( ... ) from synonym cache before saving
    for disease_key in list(DISEASE_SYNONYM_CACHE.keys()):
        cleaned_list = []

        for d in DISEASE_SYNONYM_CACHE[disease_key]:
            # Remove parentheses and content
            d = re.sub(r"\(.*?\)", "", d)

            # Normalize whitespace
            d = " ".join(d.split()).strip()

            if d:
                cleaned_list.append(d.lower())

        # Deduplicate
        DISEASE_SYNONYM_CACHE[disease_key] = sorted(set(cleaned_list))

    with open("disease_synonym_cache.json", "w", encoding="utf-8") as f:
        json.dump(DISEASE_SYNONYM_CACHE, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Search NCT, PubMed, and classify drug–disease evidence"
    )

    # named arguments (preferred)
    parser.add_argument("--input", help="Input TSV file")
    parser.add_argument("--output", help="Output GPT tsv file")
    parser.add_argument("--prompt", help="Prompt template file")

    # positional arguments (backward compatibility)
    parser.add_argument("pos_input", nargs="?", help="Input TSV file")
    parser.add_argument("pos_output", nargs="?", help="Output GPT tsv file")
    parser.add_argument("pos_prompt", nargs="?", help="Prompt template file")

    args = parser.parse_args()

    input_tsv = args.input or args.pos_input
    output_jsonl = args.output or args.pos_output
    prompt_file = args.prompt or args.pos_prompt

    if not input_tsv or not output_jsonl or not prompt_file:
        parser.error(
            "Missing arguments.\n\n"
            "Examples:\n"
            "  python 1.search_nct.py disease.tsv out.tsv prompt.txt\n"
            "  python 1.search_nct.py --input disease.tsv --output out.tsv --prompt prompt.txt"
        )

    main(input_tsv, output_jsonl, prompt_file)
