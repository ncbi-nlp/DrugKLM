#!/usr/bin/env python3
"""
Run GPT to predict mechanistic_confidence_level and mechanistic_confidence_score
based on GSEA summaries (up/down regulated pathways).

Usage:
python 6.2.KGxLM.disease2drug.prediction.mechanistic.py \
  --summary_file output/LINCS_enrich_summary/glioblastoma.GSEA.summary.jsonl \
  --prompt_file prompts/GPT4DrugGene.Final.mechanistic.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/final_mechanistic/glioblastoma
"""

import os
import json
import argparse
from collections import defaultdict
from openai import AzureOpenAI
import re

# --------------------------
# Utilities
# --------------------------

def normalize(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())
    
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_param_file(path):
    """Read tab-separated parameter file (key<TAB>value)."""
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            key, val = line.strip().split("\t", 1)
            params[key] = val
    return params

def init_azure_client(param_file):
    params = read_param_file(param_file)
    return AzureOpenAI(
        api_version=params["API_VERSION"],
        azure_endpoint=params["AZURE_OPENAI_ENDPOINT"],
        api_key=params["API_KEY"]
    ), params["DEPLOYMENT_NAME"]

def ask_gpt(client, deployment, system_prompt, user_prompt, max_tokens=2048):
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_completion_tokens=max_tokens
    )
    return resp.choices[0].message.content

def clean_json_output(output_text: str) -> str:
    """Remove markdown fences and return clean JSON text."""
    text = output_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_file", required=True, help="Path to melanoma.GSEA.summary.jsonl")
    ap.add_argument("--prompt_file", required=True, help="Prompt template for mechanistic confidence")
    ap.add_argument("--param_file", required=True, help="Azure API parameter file (tab-separated)")
    ap.add_argument("--out_dir", required=True, help="Output directory for mechanistic predictions")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompt_template = read_text(args.prompt_file)
    client, deployment = init_azure_client(args.param_file)

    # group summaries by (disease, drug)
    grouped = defaultdict(list)
    with open(args.summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            disease = entry.get("disease", "UnknownDisease")
            drug = entry.get("drug", "UnknownDrug")
            direction = entry.get("direction", "")
            pathways = entry.get("selected_pathways", [])

            grouped[(disease, drug)].append({
                "Direction": direction,
                "Pathways": pathways
            })

    # process each drug
    for (disease, drug), summaries in grouped.items():
        if not summaries:
            print(f"[SKIP] {drug} (no summaries)")
            continue

        input_block = json.dumps({"gsea_summaries": summaries}, indent=2)
        user_prompt = (
            prompt_template
            .replace("[Disease]", disease)
            .replace("[Drug]", drug)
            .replace("[input]", input_block)
        )

        try:
            system_prompt = "You are an oncology pharmacology expert specializing in mechanistic evidence synthesis."
            output_text = ask_gpt(client, deployment, system_prompt, user_prompt)

            # clean and parse
            output_text = clean_json_output(output_text)
            try:
                parsed = json.loads(output_text)
            except Exception:
                parsed = {"raw_text": output_text}

            out_path = os.path.join(args.out_dir, f"{normalize(disease)}.{drug}.final.mechanistic.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print(f"[OK] Saved {out_path}")
        except Exception as e:
            print(f"[ERROR] {drug}: {e}")

if __name__ == "__main__":
    main()

""" 
python 4.5.KGxLM.disease2drug.prediction.mechanistic.py \
  --summary_file output/LINCS_enrich_summary/glioblastoma.GSEA.summary.jsonl \
  --prompt_file prompts/4.5.GSEA_Summarizer.mechanistic.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/LINCS_enrich_summary_mechanistic/glioblastoma
  
python 4.5.KGxLM.disease2drug.prediction.mechanistic.py \
  --summary_file output/LINCS_enrich_summary/mcrc.GSEA.summary.jsonl \
  --prompt_file prompts/4.5.GSEA_Summarizer.mechanistic.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/LINCS_enrich_summary_mechanistic/mcrc

python 4.5.KGxLM.disease2drug.prediction.mechanistic.py \
  --summary_file output/LINCS_enrich_summary/melanoma.GSEA.summary.jsonl \
  --prompt_file prompts/4.5.GSEA_Summarizer.mechanistic.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/LINCS_enrich_summary_mechanistic/melanoma
"""
