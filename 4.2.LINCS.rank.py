#!/usr/bin/env python3
import argparse
import math
import pandas as pd
import json
import re
import numpy as np

def normalize_drug_name(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def load_candidate_drugs(jsonl_path: str) -> set[str]:
    drugs = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            drug = str(obj.get("Drug", "")).strip()
            if drug:
                drugs.add(normalize_drug_name(drug))
    return drugs

def parse_dose(signature: str):
    m = re.search(r'_([\d.]+)([un]M)', signature)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "nm":
        value /= 1000.0  # 轉 μM
    return value

def compute_scores(df, drug, k):
    subset = df[df["Drug"] == drug]
    if subset.empty:
        return pd.DataFrame()

    gene_stats = {}
    for _, row in subset.iterrows():
        signature = str(row.get("Signature", ""))
        dose = parse_dose(signature)

        # IC50 權重
        ln_ic50 = row.get("LN_IC50", np.nan)
        try:
            w_ic50 = math.exp(-float(ln_ic50)) if not pd.isna(ln_ic50) else 1.0
        except Exception:
            w_ic50 = 1.0
        w_ic50 = min(w_ic50, 1.0)

        # Dose 權重
        try:
            w_dose = math.exp(-k * math.log1p(dose)) if dose is not None else 1.0
        except Exception:
            w_dose = 1.0
        w_dose = min(w_dose, 1.0)

        direction = row["Direction"]
        genes = str(row["Genes"]).split(",")
        for g in genes:
            g = g.strip()
            if not g:
                continue
            if g not in gene_stats:
                gene_stats[g] = {
                    "up": 0, "down": 0,
                    "ic50_up": [], "ic50_down": [],
                    "dose_up": [], "dose_down": []
                }
            if direction == "up":
                gene_stats[g]["up"] += 1
                gene_stats[g]["ic50_up"].append(w_ic50)
                gene_stats[g]["dose_up"].append(w_dose)
            elif direction == "down":
                gene_stats[g]["down"] += 1
                gene_stats[g]["ic50_down"].append(w_ic50)
                gene_stats[g]["dose_down"].append(w_dose)

    results = []
    for gene, stats in gene_stats.items():
        total = stats["up"] + stats["down"]
        if total == 0:
            continue

        ic50_score = (np.sum(stats["ic50_up"]) - np.sum(stats["ic50_down"])) / total
        dose_score = (np.sum(stats["dose_up"]) - np.sum(stats["dose_down"])) / total

        results.append({
            "Drug": drug,
            "Gene": gene,
            "UpCount": stats["up"],
            "DownCount": stats["down"],
            "IC50Score": ic50_score,
            "IC50Score_abs": abs(ic50_score),
            "DoseScore": dose_score,
            "DoseScore_abs": abs(dose_score)
        })

    return pd.DataFrame(results)

def hybrid_rank(df, top_n, allowed_drugs, dose_weight, ic50_weight, k):
    all_results = []

    for drug in sorted(df["Drug"].unique()):
        if normalize_drug_name(drug) not in allowed_drugs:
            continue
        gene_df = compute_scores(df, drug, k)
        if not gene_df.empty:
            all_results.append(gene_df)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)

    ic50_max = df_all["IC50Score"].abs().max()
    dose_max = df_all["DoseScore"].abs().max()
    df_all["IC50Score_norm"] = df_all["IC50Score"] / ic50_max
    df_all["DoseScore_norm"] = df_all["DoseScore"] / dose_max

    df_all["FinalScore"] = (
        dose_weight * df_all["DoseScore_norm"] +
        ic50_weight * df_all["IC50Score_norm"]
    )

    results_summary = []
    results_detailed = []

    for drug, group in df_all.groupby("Drug"):
        # 依 FinalScore 分組
        group_up = group[group["FinalScore"] > 0].sort_values(
            by="FinalScore", ascending=False
        )
        group_down = group[group["FinalScore"] < 0].sort_values(
            by="FinalScore", ascending=True
        )

        # 取 top_n，如果不足就全部
        top_up = group_up.head(top_n) if len(group_up) > 0 else pd.DataFrame()
        top_down = group_down.head(top_n) if len(group_down) > 0 else pd.DataFrame()

        results_summary.append({
            "Drug": drug,
            "TopUpGenes": ",".join(top_up["Gene"].tolist()),
            "TopDownGenes": ",".join(top_down["Gene"].tolist())
        })

        results_detailed.append(pd.concat([group_up, group_down]))

    df_summary = pd.DataFrame(results_summary)
    df_detailed = pd.concat(results_detailed, ignore_index=True)
    return df_summary, df_detailed


def main():
    parser = argparse.ArgumentParser(description="Re-rank genes using IC50 and dose weighted scores.")
    parser.add_argument("--input", required=True, help="Input TSV file")
    parser.add_argument("--output", required=True, help="Output ranked genes (summary)")
    parser.add_argument("--detailed_output", required=True, help="Detailed output per-gene scores")
    parser.add_argument("--candidate_json", required=True, help="Candidate JSONL file")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--dose_weight", type=float, default=0.5, help="Weight for DoseScore")
    parser.add_argument("--ic50_weight", type=float, default=0.5, help="Weight for IC50Score")
    parser.add_argument("--dose_k", type=float, default=0.5)
    args = parser.parse_args()

    allowed_drugs = load_candidate_drugs(args.candidate_json)
    df = pd.read_csv(args.input, sep="\t")

    df_summary, df_detailed = hybrid_rank(df, args.top_n, allowed_drugs, args.dose_weight, args.ic50_weight, args.dose_k)
    df_summary.to_csv(args.output, sep="\t", index=False)
    if not df_detailed.empty:
        df_detailed.to_csv(args.detailed_output, sep="\t", index=False)

if __name__ == "__main__":
    main()


"""
python 4.2.LINCS.rank.py \
  --input output/LINCS/LINCS.search.glioblastoma.cell2gene.tsv \
  --candidate_json output/disease2drug/glioblastoma.disease2drug.candidate.jsonl \
  --output output/LINCS/LINCS.search.glioblastoma.rank.tsv \
  --detailed_output output/LINCS/LINCS.search.glioblastoma.rank.detailed.tsv \
  --top_n 200 \
  --dose_weight 0.2 \
  --ic50_weight 0.8 \
  --dose_k 0.5

python 4.2.LINCS.rank.py \
  --input output/LINCS/melanoma/LINCS.search.melanoma.cell2gene.tsv \
  --candidate_json output/disease2drug/melanoma.disease2drug.candidate.jsonl \
  --output output/LINCS/melanoma/LINCS.search.melanoma.rank.tsv \
  --detailed_output output/LINCS/melanoma/LINCS.search.melanoma.rank.detailed.tsv \
  --top_n 200 \
  --dose_weight 0.2 \
  --ic50_weight 0.8 \
  --dose_k 0.5

python 4.2.LINCS.rank.py \
  --input output/LINCS/LINCS.search.breastcancer.cell2gene.tsv \
  --candidate_json output/disease2drug/breastcancer.disease2drug.candidate.jsonl \
  --output output/LINCS/LINCS.search.breastcancer.rank.tsv \
  --detailed_output output/LINCS/LINCS.search.breastcancer.rank.detailed.tsv \
  --top_n 200 \
  --dose_weight 0.2 \
  --ic50_weight 0.8 \
  --dose_k 0.5
 
python 4.2.LINCS.rank.py \
  --input output/LINCS/mcrc/LINCS.search.mcrc.cell2gene.tsv \
  --candidate_json output/disease2drug/mcrc.disease2drug.candidate.jsonl \
  --output output/LINCS/mcrc/LINCS.search.mcrc.rank.tsv \
  --detailed_output output/LINCS/mcrc/LINCS.search.mcrc.rank.detailed.tsv \
  --top_n 200 \
  --dose_weight 0.2 \
  --ic50_weight 0.8 \
  --dose_k 0.5
"""