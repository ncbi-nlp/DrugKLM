#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Dict
import requests


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
    required = ["API_KEY", "API_VERSION", "AZURE_OPENAI_ENDPOINT", "DEPLOYMENT_NAME"]
    missing = [k for k in required if not kv.get(k)]
    if missing:
        raise ValueError(f"Missing Azure keys in param file: {missing}")
    return kv


def azure_chat_complete(params: Dict[str, str],
                        prompt_text: str,
                        retries: int = 3,
                        delay_sec: float = 2.0,
                        temperature: float = 0.0,
                        max_tokens: int = 1024,
                        allow_empty: bool = False) -> str:
    url = (
        params["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        + f"/openai/deployments/{params['DEPLOYMENT_NAME']}/chat/completions"
        + f"?api-version={params['API_VERSION']}"
    )
    headers = {"Content-Type": "application/json", "api-key": params["API_KEY"]}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs plain text only."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "n": 1,
    }

    for i in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(f"Azure error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"].get("content", None)

            # Normalize to string (or empty string) for safe handling
            content_str = (content or "").strip()

            # If empty content is acceptable, return "" instead of raising
            if content_str == "":
                if allow_empty:
                    return ""
                raise RuntimeError("Empty response content")

            return content_str
        except Exception:
            if i >= retries:
                raise
            import time
            time.sleep(delay_sec * (2 ** i))
    raise RuntimeError("Unreachable")


def main():
    ap = argparse.ArgumentParser(description="Generate subtype statement and optional subtype list for a disease using GPT")
    ap.add_argument("--disease", required=True)
    ap.add_argument("--prompt", required=True, help="Prompt template for generating subtype statement")
    ap.add_argument("--param_file", required=True, help="Azure parameter file")
    ap.add_argument("--output", required=True, help="Output txt file for full statement")
    ap.add_argument("--output_list", required=False, help="Output txt file for subtype list")
    ap.add_argument("--list_prompt", required=False, help="Prompt template file for generating subtype list")
    args = ap.parse_args()

    if os.path.isfile(args.disease) and args.disease.lower().endswith(".json"):
        with open(args.disease, "r", encoding="utf-8") as jf:
            json_input = jf.read().strip()
        with open(args.prompt, "r", encoding="utf-8") as f:
            template = f.read()
        prompt_text = template.replace("[JSON_Input]", json_input)
        disease = os.path.splitext(os.path.basename(args.disease))[0]  # e.g. melanoma
    
    print(prompt_text)

    params = read_params_tsv(args.param_file)
    result = azure_chat_complete(params, prompt_text)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    norm_stmt = (result or "").strip()

    with open(args.output, "w", encoding="utf-8") as fout:
        # If the statement is empty (""), literal '""', or 'N/A', don't write the header
        if norm_stmt in ("", '""') or norm_stmt.upper() == "N/A":
            # Choose ONE of the following two behaviors based on your downstream expectations:
            fout.write("")   # OR write a truly empty file
        else:
            fout.write("You must also identify which " + disease + " subtype the evidence most strongly supports, restricted to the following seven:\n")
            fout.write(norm_stmt)

    sys.stderr.write(f"[OK] Subtype statement saved to {args.output}\n")

    # Generate subtype list only if output_list and list_prompt are specified
    if args.output_list and args.list_prompt:
        # Normalize the first-call result
        norm_result = (result or "").strip()

        # If your pipeline uses either N/A or the refined-prompt empty-string rule, skip second call
        if norm_result.upper() == "N/A" or norm_result == '""' or norm_result == "":
            sys.stderr.write("[INFO] Subtype statement indicates no further subtypes; skipping subtype list generation.\n")
            os.makedirs(os.path.dirname(args.output_list) or ".", exist_ok=True)
            # If your downstream expects literal empty string token, write "" (two quotes). Otherwise, write blank.
            with open(args.output_list, "w", encoding="utf-8") as fout:
                # Choose one of the two lines below based on your convention:
                fout.write("")   # or write truly empty file
        else:
            with open(args.list_prompt, "r", encoding="utf-8") as f:
                list_template = f.read()
            list_prompt_text = list_template.replace("[STATEMENT]", result)
            # Allow empty in case the list prompt also decides it's too granular.
            list_result = azure_chat_complete(params, list_prompt_text, allow_empty=True)

            os.makedirs(os.path.dirname(args.output_list) or ".", exist_ok=True)
            with open(args.output_list, "w", encoding="utf-8") as fout:
                # If the model returned an empty string, emit your canonical marker
                if (list_result or "").strip() == "":
                    fout.write("")
                else:
                    fout.write(list_result)
            sys.stderr.write(f"[OK] Subtype list saved to {args.output_list}\n")


if __name__ == "__main__":
    main()


"""
python 6.0.KGxLM.disease2drug.subtype_generation.py \
  --disease input/melanoma.json \
  --prompt prompts/6.0.SubtypeGeneration.txt \
  --list_prompt prompts/6.0.SubtypelistGeneration.txt \
  --param_file parameter.gpt4o.txt \
  --output prompts/subtype/melanoma.subtype.statement.txt \
  --output_list prompts/subtype/melanoma.subtype.txt

python 6.0.KGxLM.disease2drug.subtype_generation.py \
  --disease input/mCRC.json \
  --prompt prompts/6.0.SubtypeGeneration.txt \
  --list_prompt prompts/6.0.SubtypelistGeneration.txt \
  --param_file parameter.gpt4o.txt \
  --output prompts/subtype/mcrc.subtype.statement.txt \
  --output_list prompts/subtype/mcrc.subtype.txt
"""
