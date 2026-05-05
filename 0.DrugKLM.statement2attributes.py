#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statement -> Attributes extractor with OpenAI / Azure OpenAI + MedCPT disease verification.

- Reads params from TSV (prefer 'parameter.gpt4o.txt' per your workflow).
- Auto-detects backend: if OPENAI_API_KEY is set in the param file, uses
  the public OpenAI Chat Completions API; otherwise falls back to Azure.
- Builds prompt from template with inserted statement.
- Extracts the first balanced JSON from the response.
- Optionally validates disease name via MedCPT embeddings:
  If top-1 cosine score != 1.000 (rounded to 3 decimals), append "disease_match".

Param file (TSV, key<TAB>value) for OpenAI mode:
    OPENAI_API_KEY    sk-...
    OPENAI_MODEL      gpt-4o          # or gpt-4o-mini, gpt-4-turbo, etc.
    OPENAI_BASE_URL   https://api.openai.com/v1   # optional override

Param file for Azure mode (legacy, unchanged):
    AZURE_OPENAI_ENDPOINT     https://<resource>.openai.azure.com/
    API_KEY                   <azure key>
    DEPLOYMENT_NAME           gpt-4o
    API_VERSION               2024-05-01-preview
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import numpy as np


# -------------------------------
# Utilities and I/O
# -------------------------------

def read_params_tsv(path: str) -> Dict[str, str]:
    """
    Read a 2-column TSV of key\tvalue into a dict.

    Auto-detects backend:
      * If OPENAI_API_KEY (or API_KEY without an Azure endpoint) is present,
        runs in OpenAI mode and requires OPENAI_API_KEY + OPENAI_MODEL.
      * Otherwise, runs in Azure mode and requires the four AZURE_OPENAI_* fields.

    Returns a normalized dict that always contains a 'BACKEND' key set to
    either 'openai' or 'azure', plus the canonical credential fields for
    that backend.
    """
    d: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[0].strip(), parts[1].strip()
            d[k] = v

    # Decide backend.
    has_openai_key = bool(d.get("OPENAI_API_KEY"))
    has_azure_endpoint = bool(d.get("AZURE_OPENAI_ENDPOINT") or d.get("ENDPOINT"))
    backend = "openai" if (has_openai_key and not has_azure_endpoint) else "azure"

    norm: Dict[str, str] = {"BACKEND": backend}

    if backend == "openai":
        api_key = d.get("OPENAI_API_KEY") or d.get("API_KEY")
        model = d.get("OPENAI_MODEL") or d.get("MODEL") or d.get("DEPLOYMENT_NAME")
        base_url = d.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        missing = [k for k, v in [("OPENAI_API_KEY", api_key), ("OPENAI_MODEL", model)] if not v]
        if missing:
            raise ValueError(
                f"OpenAI backend selected but missing fields: {missing}. "
                f"Param file must contain OPENAI_API_KEY and OPENAI_MODEL."
            )
        norm["OPENAI_API_KEY"] = api_key
        norm["OPENAI_MODEL"] = model
        norm["OPENAI_BASE_URL"] = base_url
        return norm

    # Azure mode (default / legacy)
    aliases = {
        "AZURE_OPENAI_ENDPOINT":  ["AZURE_OPENAI_ENDPOINT", "ENDPOINT"],
        "AZURE_OPENAI_API_KEY":   ["AZURE_OPENAI_API_KEY", "API_KEY"],
        "AZURE_OPENAI_DEPLOYMENT":["AZURE_OPENAI_DEPLOYMENT", "DEPLOYMENT_NAME", "DEPLOYMENT"],
        "AZURE_OPENAI_API_VERSION":["AZURE_OPENAI_API_VERSION", "API_VERSION"],
    }
    for canonical, keys in aliases.items():
        for k in keys:
            if k in d and d[k]:
                norm[canonical] = d[k]
                break

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"]
    missing = [k for k in required if k not in norm or not norm[k]]
    if missing:
        raise ValueError(f"Missing required params in TSV: {missing}. "
                         f"Accepted keys include: {aliases}")

    return norm


def build_prompt(prompt_path: str, statement: str) -> str:
    """Insert statement into the prompt template."""
    tpl = Path(prompt_path).read_text(encoding="utf-8")
    return tpl.replace("[INSERT STATEMENT HERE]", statement)


# -------------------------------
# OpenAI / Azure OpenAI call
# -------------------------------

def openai_chat_complete(
    api_key: str,
    model: str,
    prompt_text: str,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    max_completion_tokens: int = 2048,
    retries: int = 3,
    delay_sec: float = 2.0,
) -> str:
    """Call the public OpenAI Chat Completions API and return assistant content text."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs STRICT JSON only."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "n": 1,
        "response_format": {"type": "json_object"},
    }

    for i in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after is not None else delay_sec * (2 ** i)
                wait = wait + (0.25 * wait)  # jitter
                time.sleep(min(wait, 60.0))
                if i >= retries:
                    raise RuntimeError(f"OpenAI rate limit hit repeatedly (429): {resp.text[:500]}")
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                return content
            raise RuntimeError("Empty response content")
        except Exception:
            if i >= retries:
                raise
            time.sleep(min(delay_sec * (2 ** i), 60.0))


def azure_chat_complete(
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    prompt_text: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 2048,
    retries: int = 3,
    delay_sec: float = 2.0,
) -> str:
    """Call Azure OpenAI Chat Completions and return assistant content text."""
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs STRICT JSON only."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "n": 1,
        "response_format": {"type": "json_object"},
    }

    for i in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after is not None else delay_sec * (2 ** i)
                wait = wait + (0.25 * wait)  # jitter
                time.sleep(min(wait, 60.0))
                if i >= retries:
                    raise RuntimeError(f"Azure rate limit hit repeatedly (429): {resp.text[:500]}")
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Azure error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                return content
            raise RuntimeError("Empty response content")
        except Exception:
            if i >= retries:
                raise
            time.sleep(min(delay_sec * (2 ** i), 60.0))


# -------------------------------
# JSON extraction / parsing
# -------------------------------

def extract_first_json(text: str) -> Optional[str]:
    """Extract first balanced {...} JSON object using a simple bracket stack."""
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start:i+1]
    return None


def repair_json(s: str) -> str:
    """Light-weight repairs: remove trailing commas, fix common quotes."""
    # Remove BOM/zero widths
    s = s.replace("\ufeff", "").replace("\u200b", "")
    # Replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_model_json(text: str) -> Dict[str, Any]:
    """Parse model output into dict, with first-JSON extraction and repair."""
    # Strip common markdown fences first
    clean = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.MULTILINE)
    cand = extract_first_json(clean)
    if not cand:
        raise ValueError("No JSON object found in model output.")
    fixed = repair_json(cand)
    return json.loads(fixed)


# -------------------------------
# MedCPT lookup (lazy dependencies)
# -------------------------------

def _lazy_medcpt_imports():
    """
    Lazy import PyTorch and Transformers only if MedCPT search is requested.
    """
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModel  # noqa
    return torch, AutoTokenizer, AutoModel


def medcpt_best_match(query: str, embeddings_npy: str, index_tsv: str | None = None) -> tuple[str, float]:
    """
    Return (best_text, best_score) for disease query using MedCPT.
    It expects embeddings_npy to be a dict: key -> 1D vector (numpy).
    """
    if not query or not query.strip():
        return "", 0.0

    if not os.path.exists(embeddings_npy):
        raise FileNotFoundError(f"MedCPT embeddings not found: {embeddings_npy}")
    data = np.load(embeddings_npy, allow_pickle=True).item()
    if not isinstance(data, dict) or not data:
        raise ValueError("MedCPT embeddings must be a non-empty dict of key -> vector.")

    keys = list(data.keys())
    vecs = [np.asarray(data[k], dtype=np.float32) for k in keys]
    dim = vecs[0].shape[0]
    for i, v in enumerate(vecs):
        if v.ndim != 1 or v.shape[0] != dim:
            raise ValueError(f"Inconsistent vector shape for key '{keys[i]}': {v.shape}")
    mat = np.vstack(vecs).astype(np.float32)  # (N, D)

    key2text = {k: k for k in keys}
    if index_tsv and os.path.exists(index_tsv):
        with open(index_tsv, "r", encoding="utf-8") as fh:
            _ = fh.readline()  # skip header if present
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    k, txt = parts[0], parts[1]
                    key2text[k] = txt

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_norm = mat / norms

    torch, AutoTokenizer, AutoModel = _lazy_medcpt_imports()
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")

    inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        q = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)

    scores = mat_norm @ qn  # cosine similarity
    top = int(np.argmax(scores))
    best_key = keys[top]
    best_score = float(scores[top])
    best_text = key2text.get(best_key, best_key)
    return best_text, best_score


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="Path to input clinical statement text, or '-' for stdin")
    ap.add_argument("--param_file", type=str, required=True, help="TSV with Azure params")
    ap.add_argument("--prompt", type=str, required=True, help="Prompt template path")
    ap.add_argument("--out", type=str, required=False, help="Optional output JSON path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_completion_tokens", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=None, help="Backward-compatible alias")

    # MedCPT options
    ap.add_argument("--medcpt_embeddings", type=str, required=False,
                    help="Path to MedCPT disease embeddings .npy (dict: key -> vector)")
    ap.add_argument("--medcpt_index", type=str, required=False,
                    help="Optional TSV that maps key -> display text for embeddings")

    args = ap.parse_args()

    # Read inputs
    if args.infile == "-":
        statement = sys.stdin.read().strip()
    else:
        statement = Path(args.infile).read_text(encoding="utf-8").strip()

    params = read_params_tsv(args.param_file)
    prompt_text = build_prompt(args.prompt, statement)

    # Azure call
    max_completion_tokens = args.max_completion_tokens
    if args.max_tokens is not None:
        max_completion_tokens = args.max_tokens

    if params.get("BACKEND") == "openai":
        content = openai_chat_complete(
            api_key=params["OPENAI_API_KEY"],
            model=params["OPENAI_MODEL"],
            prompt_text=prompt_text,
            base_url=params.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=args.temperature,
            max_completion_tokens=max_completion_tokens,
        )
    else:
        content = azure_chat_complete(
            endpoint=params["AZURE_OPENAI_ENDPOINT"],
            api_key=params["AZURE_OPENAI_API_KEY"],
            deployment=params["AZURE_OPENAI_DEPLOYMENT"],
            api_version=params["AZURE_OPENAI_API_VERSION"],
            prompt_text=prompt_text,
            temperature=args.temperature,
            max_completion_tokens=max_completion_tokens,
        )

    # Parse model JSON
    
    parsed = parse_model_json(content)

    # Build result
    result: Dict[str, Any] = {
        "disease": parsed.get("disease"),
        "stage": parsed.get("stage"),
        "variant": parsed.get("variant"),
        "metastasis": parsed.get("metastasis"),
        "used_drugs": parsed.get("used_drugs"),
        "clinical_history_summary": parsed.get("clinical_history_summary"),
        "overall_summary": parsed.get("overall_summary")
    }

    # MedCPT verification: attach disease_match if top-1 score != 1.000
    if args.medcpt_embeddings and result.get("disease"):
        try:
            best_text, best_score = medcpt_best_match(
                query=result["disease"],
                embeddings_npy=args.medcpt_embeddings,
                index_tsv=args.medcpt_index
            )
            if float(f"{best_score:.3f}") != 1.000:
                result["disease_match"] = best_text
            else:
                result["disease_match"] = result.get("disease")
        except Exception:
            # Fail-soft: ignore MedCPT errors
            pass

    # Output
    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(pretty, encoding="utf-8")
    print(pretty)


if __name__ == "__main__":
    main()

"""
python 0.KGxLM.statement2attributes.py \
  --infile input/ex.txt \
  --param_file parameter.gpt4o.txt \
  --prompt prompts/0.Statement2Attributes.txt \
  --medcpt_embeddings MedCPT.npy/disease_embeddings.npy \
  --medcpt_index MedCPT.npy/disease_index.tsv \
  --out input/ex.json

"""
