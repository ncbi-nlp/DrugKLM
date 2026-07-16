#!/usr/bin/env python3
import os
import json
import csv
import argparse

LEVEL_ORDER = {
    "Very High": 5,
    "High": 4,
    "Moderately High": 3,
    "Moderate": 2,
    "Moderately Low": 1,
    "Low": 0,
    "Very Low": -1
}

def convert_jsons_to_tsv(input_folder, output_tsv):
    rows = []
    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            file_path = os.path.join(input_folder, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list) and data:
                data = data[0]

            disease = data.get("Disease", "")
            drug = data.get("Drug", "")
            verdict = data.get("verdict", {})
            rationale = "; ".join(data.get("rationale_bullets", []))
            top_genes = "; ".join([f"{g['gene']}({g['mechanism']})" for g in data.get("top_genes", [])])
            risks = "; ".join(data.get("risks", []))
            next_steps = "; ".join(data.get("next_steps", []))

            rows.append({
                "disease": disease,
                "drug": drug,
                "overall_confidence_level": verdict.get("overall_confidence_level", ""),
                "subtype_confidence_level": verdict.get("subtype_confidence_level", ""),
                "subtype": verdict.get("subtype", ""),
                "risk_level": verdict.get("risk_level", ""),
                "overall_confidence_score": verdict.get("overall_confidence_score", 0),
                "subtype_confidence_score": verdict.get("subtype_confidence_score", 0),
                "risk_score": verdict.get("risk_score", 0),
                "FDA_status": verdict.get("FDA_status", ""),
                "rationale_bullets": rationale,
                "top_genes": top_genes,
                "risks": risks,
                "next_steps": next_steps
            })

    # 排序規則不變
    rows.sort(
        key=lambda r: (
            int(r["overall_confidence_score"]) if str(r["overall_confidence_score"]).isdigit() else 0,
            int(r["subtype_confidence_score"]) if str(r["subtype_confidence_score"]).isdigit() else 0,
            -(int(r["risk_score"]) if str(r["risk_score"]).isdigit() else 9999)
        ),
        reverse=True
    )

    fieldnames = [
        "disease", "drug", "overall_confidence_level", "subtype_confidence_level",
        "subtype", "risk_level",
        "overall_confidence_score", "subtype_confidence_score",
        "risk_score", "FDA_status", "rationale_bullets", "top_genes", "risks", "next_steps"
    ]

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    with open(output_tsv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"TSV saved to {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON prediction results to TSV.")
    parser.add_argument("--input_folder", required=True, help="Input folder containing .json files")
    parser.add_argument("--output_tsv", required=True, help="Output TSV file path")
    args = parser.parse_args()

    convert_jsons_to_tsv(args.input_folder, args.output_tsv)

"""
python 7.KGxLM.disease2drug.json2tsv.py \
  --input_folder output/final_prediction/case_mcrc \
  --output_tsv output/case_mcrc.final_prediction.tsv
  
python 7.KGxLM.disease2drug.json2tsv.py \
  --input_folder output/final_prediction/case_melanoma \
  --output_tsv output/case_melanoma.final_prediction.tsv
"""