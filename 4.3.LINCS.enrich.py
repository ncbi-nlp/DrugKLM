#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import sys

# 假設你有一個 enrich() 和 save_enrichment() 可用
# 如果這些在 utils 或 gsea 包裡，請自行 import
from gseapy import enrichr

def enrich(genes, species, dbs, p_cutoff):
    """呼叫 Enrichr 做基因集富集分析"""
    return enrichr(gene_list=genes, gene_sets=dbs, organism=species, cutoff=p_cutoff, outdir=None)

def save_enrichment(enr, out_file):
    """保存 enrichment 結果到 TSV"""
    if enr is None or len(enr.results) == 0:
        pd.DataFrame().to_csv(out_file, sep="\t", index=False)
    else:
        enr.results.to_csv(out_file, sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser(description="Run enrichment for LINCS ranked genes (new format)")
    parser.add_argument("--input", required=True, help="Input .rank.tsv from 4.2.LINCS.rank.py")
    parser.add_argument("--species", default="9606", help="Species (default: human, 9606)")
    parser.add_argument("--dbs", nargs="+", required=True, help="List of enrichment libraries (e.g. KEGG_2021_Human GO_Biological_Process_2021)")
    parser.add_argument("--p_cutoff", type=float, default=0.05, help="P-value cutoff")
    parser.add_argument("--outdir", required=True, help="Output directory for GSEA results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")

    for _, row in df.iterrows():
        drug = str(row["Drug"]).strip()

        # 1) 上調基因
        if "TopUpGenes" in row and isinstance(row["TopUpGenes"], str) and row["TopUpGenes"].strip():
            genes_up = [g.strip() for g in row["TopUpGenes"].split(",") if g.strip()]
            out_file_up = outdir / f"{drug}.up.GSEA.tsv"
            try:
                gsea_up = enrich(sorted(genes_up), args.species, args.dbs, args.p_cutoff)
                save_enrichment(gsea_up, out_file_up)
                print(f"[OK] {drug} up → {out_file_up}")
            except Exception as e:
                print(f"[WARN] enrichment failed for {drug} up: {e}", file=sys.stderr)
                save_enrichment(None, out_file_up)

        # 2) 下調基因
        if "TopDownGenes" in row and isinstance(row["TopDownGenes"], str) and row["TopDownGenes"].strip():
            genes_down = [g.strip() for g in row["TopDownGenes"].split(",") if g.strip()]
            out_file_down = outdir / f"{drug}.down.GSEA.tsv"
            try:
                gsea_down = enrich(sorted(genes_down), args.species, args.dbs, args.p_cutoff)
                save_enrichment(gsea_down, out_file_down)
                print(f"[OK] {drug} down → {out_file_down}")
            except Exception as e:
                print(f"[WARN] enrichment failed for {drug} down: {e}", file=sys.stderr)
                save_enrichment(None, out_file_down)

if __name__ == "__main__":
    main()


"""
python 4.3.LINCS.enrich.py \
  --input output/LINCS/glioblastoma/LINCS.search.glioblastoma.rank.tsv \
  --species Human \
  --dbs GO_Biological_Process_2021 KEGG_2021_Human \
  --p_cutoff 0.05 \
  --outdir output/LINCS_enrich/glioblastoma
  
python 4.3.LINCS.enrich.py \
  --input output/LINCS/melanoma/LINCS.search.melanoma.rank.tsv \
  --species Human \
  --dbs GO_Biological_Process_2021 KEGG_2021_Human \
  --p_cutoff 0.05 \
  --outdir output/LINCS_enrich/melanoma
 
python 4.3.LINCS.enrich.py \
  --input output/LINCS/mcrc/LINCS.search.mcrc.rank.tsv \
  --species Human \
  --dbs GO_Biological_Process_2021 KEGG_2021_Human \
  --p_cutoff 0.05 \
  --outdir output/LINCS_enrich/mcrc
"""