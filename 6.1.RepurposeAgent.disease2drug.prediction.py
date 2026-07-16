#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference on pre-summarized input payloads using GPT4DrugGene.Final.txt prompt.
Each *.input.json in --input_dir will be loaded and sent to GPT.
Results are saved as one JSON per drug in --out_dir.
"""

import os
import json
import argparse
from pathlib import Path
import openai

# ---------------------------
# Azure OpenAI helpers
# ---------------------------

def init_azure_client(param_file: str):
    """
    Initialize Azure OpenAI client using a param TSV file with keys:
    AZURE_OPENAI_ENDPOINT, API_VERSION, DEPLOYMENT_NAME, API_KEY
    """
    params = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.rstrip("\n").split("\t", 1)
            params[k] = v

    endpoint = params["AZURE_OPENAI_ENDPOINT"]
    version  = params["API_VERSION"]
    deploy   = params["DEPLOYMENT_NAME"]
    key      = params["API_KEY"]

    openai.api_type = "azure"
    openai.api_base = endpoint
    openai.api_key = key
    openai.api_version = version

    client = openai.AzureOpenAI(api_key=key, api_version=version, azure_endpoint=endpoint)
    return client, deploy

def truncate_incomplete_json_array(text: str) -> str:
    """
    Attempt to salvage partial JSON objects from a GPT output string.
    Returns a JSON array string containing all successfully decoded objects.
    """
    dec = json.JSONDecoder()
    i = 0
    objects = []
    while True:
        start = text.find("{", i)
        if start == -1:
            break
        try:
            obj, end = dec.raw_decode(text, start)
            if isinstance(obj, dict):
                objects.append(obj)
            i = end
        except json.JSONDecodeError:
            i = start + 1
    if not objects:
        raise ValueError("No complete JSON objects found in input.")
    return json.dumps(objects, ensure_ascii=False, indent=2)

# ---------------------------
# Prompt + GPT caller
# ---------------------------

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ask_gpt(client, deployment: str, system_prompt: str, user_prompt: str) -> str:
    """
    Call Azure GPT chat API and return model output text.
    """
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_completion_tokens=4096
    )
    return resp.choices[0].message.content

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate final predictions using pre-summarized input JSONs and GPT prompt")
    ap.add_argument("--input_dir", required=True, help="Directory containing *.input.json payloads")
    ap.add_argument("--prompt_file", required=True, help="Prompt template (e.g., GPT4DrugGene.Final.txt)")
    ap.add_argument("--param_file", required=True, help="Azure parameter TSV file")
    ap.add_argument("--out_dir", required=True, help="Directory to save GPT outputs")
    ap.add_argument("--subtypes_file", required=True, help="Path to melanoma.subtype.txt")
    ap.add_argument("--subtype_statements_file", required=True, help="Path to melanoma.subtype.statement.txt")
    ap.add_argument("--json_input", required=True, help="Path to case-level JSON (e.g., input/melanoma.json)")  # NEW
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompt_tpl = load_prompt_template(args.prompt_file)

    # Read melanoma subtype info
    with open(args.subtypes_file, "r", encoding="utf-8") as f:
        subtypes = f.read().strip()
    with open(args.subtype_statements_file, "r", encoding="utf-8") as f:
        subtype_statements = f.read().strip()

    # Read the full case-level JSON blob for [JSON_Input]
    with open(args.json_input, "r", encoding="utf-8") as jf:
        json_input_blob = jf.read().strip()

    # Parse case JSON once so we can inject it into every output file
    try:
        case_json_obj = json.loads(json_input_blob)
    except Exception:
        # Fallback: keep raw text if parsing fails
        case_json_obj = {"raw_text": json_input_blob}

    # Prepare a base template with static injections (used for every drug)
    _subtype_statements_norm = (subtype_statements or "").strip()
    if _subtype_statements_norm == "" or _subtype_statements_norm == '""':
        _subtype_statements_injection = "Subtypes: N/A (no recognized subtypes)."
    else:
        _subtype_statements_injection = subtype_statements

    prompt_tpl = (
        prompt_tpl
        .replace("[subtypes]", subtypes)
        .replace("[subtype_statements]", _subtype_statements_injection)
        .replace("[JSON_Input]", json_input_blob)  # inject entire case JSON here
    )

    client, deployment = init_azure_client(args.param_file)

    for path in sorted(Path(args.input_dir).glob("*.input.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        disease = payload.get("Disease", "")
        drug = payload.get("Drug", "")
        input_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

        user_prompt = (
            prompt_tpl
            .replace("[Disease]", disease)
            .replace("[Drug]", drug)
            .replace("[input]", input_json_str)
        )

        print(f"[INFO] Processing {drug} for {disease} ...")
        try:
            output_text = ask_gpt(
                client, deployment,
                "You are an oncology pharmacology expert specializing in evidence synthesis.",
                user_prompt
            )
            try:
                parsed = json.loads(output_text)
                out_obj = parsed
            except Exception:
                try:
                    fixed = truncate_incomplete_json_array(output_text)
                    out_obj = json.loads(fixed)
                except Exception:
                    out_obj = {"raw_text": output_text}

            # merge mechanistic verdict into output verdict
            mech = payload.get("gsea_mechanistic", {}).get("mechanistic_verdict", {})
            if mech and isinstance(out_obj, dict):
                verdict = out_obj.get("verdict", {})
                if isinstance(verdict, dict):
                    if "mechanistic_confidence_level" not in verdict:
                        verdict["mechanistic_confidence_level"] = mech.get("mechanistic_confidence_level", "")
                    if "mechanistic_confidence_score" not in verdict:
                        verdict["mechanistic_confidence_score"] = mech.get("mechanistic_confidence_score", "")
                    out_obj["verdict"] = verdict

            # --- NEW: attach the case JSON into the final output ---
            if isinstance(out_obj, dict):
                # Common case: model returns a single JSON object
                out_obj["case_json"] = case_json_obj
            elif isinstance(out_obj, list):
                # If the model returns a list, attach case_json to each dict element; wrap scalars
                new_list = []
                for item in out_obj:
                    if isinstance(item, dict):
                        item = {**item, "case_json": case_json_obj}
                    else:
                        item = {"item": item, "case_json": case_json_obj}
                    new_list.append(item)
                out_obj = new_list
            else:
                # Fallback: wrap non-JSON/other structures
                out_obj = {"result": out_obj, "case_json": case_json_obj}
            # --- end NEW ---

            out_file = Path(args.out_dir) / f"{disease}.{drug}.final.prediction.json"
            with open(out_file, "w", encoding="utf-8") as fout:
                json.dump(out_obj, fout, ensure_ascii=False, indent=2)
            
            print(f"[OK] Saved prediction: {args.out_dir}")

        except Exception as e:
            print(f"[ERR] Failed GPT call for {drug}: {e}")


if __name__ == "__main__":
    main()


"""
python 6.1.KGxLM.disease2drug.prediction.py \
  --input_dir output/final_input/glioblastoma \
  --prompt_file prompts/6.1.GPT4DrugGene.Final.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/final_prediction/glioblastoma \
  --subtypes_file prompts/subtype/glioblastoma.subtype.txt \
  --subtype_statements_file prompts/subtype/glioblastoma.subtype.statement.txt
 
python 6.1.KGxLM.disease2drug.prediction.py \
  --input_dir output/final_input/melanoma \
  --prompt_file prompts/6.1.GPT4DrugGene.Final.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/final_prediction/melanoma \
  --subtypes_file prompts/subtype/melanoma.subtype.txt \
  --subtype_statements_file prompts/subtype/melanoma.subtype.statement.txt \
  --json_input input/melanoma.json

python 6.1.KGxLM.disease2drug.prediction.py \
  --input_dir output/final_input/mcrc \
  --prompt_file prompts/6.1.GPT4DrugGene.Final.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/final_prediction/mcrc \
  --subtypes_file prompts/subtype/mcrc.subtype.txt \
  --subtype_statements_file prompts/subtype/mcrc.subtype.statement.txt \
  --json_input input/mCRC.json
"""
