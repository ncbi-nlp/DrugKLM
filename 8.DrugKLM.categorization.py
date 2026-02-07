#!/usr/bin/env python3

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

# ======================
# Azure OpenAI helpers
# ======================
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


def process_drug_disease_pair(drug, disease, drugs_by_nct, diseases_by_nct):
    drug_l = drug.lower()
    disease_l = disease.lower()

    matched_ncts = sorted(
        nct for nct in drugs_by_nct
        if any(drug_l in d.lower() for d in drugs_by_nct[nct])
        and any(disease_l in d.lower() for d in diseases_by_nct.get(nct, []))
    )

    study_cache = {}
    trials = []

    for nct_id in matched_ncts:
        study = get_study_json(nct_id, study_cache)

        phases, overall_status, has_results, last_update_posted = extract_phase_status(study)

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
                has_results,
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
    
    CATEGORIZATION_FIELDS = [
        "FDA-Approved",
        "FDA-Approved Indication",
        "Mono/Combo Therapy",
        "Category #",
        "Category Name",
        "Reference"
    ]
    
    print("Loading drugs / diseases...")
    drugs_by_nct = load_all_drugs()
    diseases_by_nct = load_all_diseases()

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

            # === 原本的 Categorization 流程（完全不動） ===
            trials = process_drug_disease_pair(
                drug, disease, drugs_by_nct, diseases_by_nct
            )

            if not trials:
                pmids = search_pubmed_by_drug_disease(drug, disease, top_n=5)
                pubmed_records = fetch_pubmed_records(pmids) if pmids else []
                evidence_obj = {
                    "input_disease": disease,
                    "input_drug": drug,
                    "pubmed": pubmed_records
                }
            else:
                evidence_obj = {
                    "input_disease": disease,
                    "input_drug": drug,
                    "trials": trials
                }

            prompt_text = build_prompt(
                prompt_template, drug, disease, evidence_obj
            )

            try:
                gpt_response = call_azure_gpt4o(prompt_text, azure_config)
                gpt_json = safe_parse_json(gpt_response)
            except Exception as e:
                gpt_json = {}

            status_cat = gpt_json.get("Status Category", "")

            # --- FDA-Approved: input FDA_status > Category semantics > GPT ---
            fda_approved = gpt_json.get("FDA-Approved", "")

            input_fda_status = row.get("FDA_status", "").lower()

            # --- fda_approved_indication: rule-based override ---
            fda_approved_indication = gpt_json.get("FDA_Approved_Indication", "")
            
            # Rule 1: input says FDA-approved (drug-level)
            if input_fda_status.startswith("fda-approved"):
                fda_approved = "Yes"

            # Rule 2: Category # = 1 implies FDA-approved for this disease
            else:
                try:
                    if int(status_cat) == 1:
                        fda_approved = "Yes"
                except Exception:
                    pass
            
            row["FDA-Approved"] = fda_approved
            row["FDA-Approved Indication"] = fda_approved_indication
            row["Mono/Combo Therapy"] = gpt_json.get("Mono/Combo Therapy", "")
            row["Category #"] = status_cat
            row["Reference"] = gpt_json.get("Reference", "")

            # ---- derive Category Name from Category # ----
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
