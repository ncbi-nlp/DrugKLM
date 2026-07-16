#!/bin/bash
###############################################################################
# RepurposeAgent.sh
#
# 	Disease-specific Drug Ranking & Mechanistic Evidence Pipeline
#
# Usage:
#   ./RepurposeAgent.sh "Disease Name"
#
# Example:
#   ./RepurposeAgent.sh "Acute Myeloid Leukemia"
#
# This script executes the full RepurposeAgent pipeline:
#   Disease → Attributes → Drug/Gene Candidates → Evidence Integration
#   → LINCS perturbation analysis → GSEA → Mechanistic summarization
#   → Disease subtype-aware drug ranking
###############################################################################

############################
# 0. Input disease setting
############################
Disease=$1
echo "Running RepurposeAgent pipeline for disease: ${Disease}"

# Save disease name as a single-line input file
echo "${Disease}" > input/${Disease}.txt


###############################################################################
# 1. Disease statement → structured disease attributes
#    - Map free-text disease name to controlled disease concepts
#    - Use MedCPT embeddings for disease normalization
###############################################################################
python 0.RepurposeAgent.statement2attributes.py \
  --infile input/${Disease}.txt \
  --param_file parameter.gpt4o.txt \
  --prompt prompts/0.Statement2Attributes.txt \
  --medcpt_embeddings MedCPT.npy/disease_embeddings.npy \
  --medcpt_index MedCPT.npy/disease_index.tsv \
  --out input/${Disease}.json


###############################################################################
# 2. Disease → Drug candidate generation (KG-based)
#    - Retrieve candidate drugs associated with the disease
#    - No ranking yet, recall-oriented
###############################################################################
python 1.RepurposeAgent.disease2drug.candidate.py \
  --json_input input/${Disease}.json \
  --output output/disease2drug/${Disease}.disease2drug.candidate.jsonl


###############################################################################
# 3. Disease → Gene candidate generation (KG-based)
#    - Retrieve disease-associated genes for downstream evidence matching
###############################################################################
python 2.RepurposeAgent.disease2gene.candidate.py \
  --json_input input/${Disease}.json \
  --output output/disease2gene/${Disease}.disease2gene.candidate.jsonl


###############################################################################
# 4. Drug–Gene evidence integration
#    - Aggregate multi-source evidence:
#        CTD, PubTator, DGIdb, LINCS
#    - GPT-based ranking of drug–gene relevance
###############################################################################
python 3.RepurposeAgent.disease2drug2gene.evidence.py \
  --input output/disease2drug/${Disease}.disease2drug.candidate.jsonl \
  --disease2gene_candidate_json output/disease2gene/${Disease}.disease2gene.candidate.jsonl \
  --ctd_tsv DB/DrugGeneRelationEvdience.CTD.tsv \
  --pubtator_tsv DB/DrugGeneRelationEvdience.PubTator3.tsv \
  --dgidb_tsv DB/DrugGeneRelationEvdience.dgidb.drug2genes.tsv \
  --lincs_tsv DB/DrugGeneRelationEvdience.LINCS.cp_coeff.tsv \
  --prompt_topk prompts/3.GPT4DrugGene.Ranking.txt \
  --disease_detail_json input/${Disease}.json \
  --output_folder output/drug2gene/${Disease}


###############################################################################
# 5. LINCS perturbation-based functional validation
###############################################################################

# Create disease-specific LINCS output folder
mkdir -p output/LINCS/${Disease}


########################################
# 5.1 LINCS cell-line & gene matching
#     - Map disease → relevant cell lines
#     - Integrate GDSC / CCLE drug response
########################################
python 4.1.LINCS.search.py \
  --input_json input/${Disease}.json \
  --mapping_jsonl DB/cell2disease.merge.LINCS.variants.jsonl \
  --gmt_folder DB/LINCS/gmt \
  --gdsc_file2 DB/GDSC/GDSC2_fitted_dose_response_27Oct23.csv \
  --gdsc_file1 DB/GDSC/GDSC1_fitted_dose_response_27Oct23.csv \
  --ccle_file DB/CCLE/DrugResponse.txt \
  --output output/LINCS/${Disease}/LINCS.search.${Disease}.cell2gene.tsv


########################################
# 5.2 LINCS drug ranking
#     - Rank drugs using perturbation strength
#     - IC50 weighted higher than dosage
########################################
python 4.2.LINCS.rank.py \
  --input output/LINCS/${Disease}/LINCS.search.${Disease}.cell2gene.tsv \
  --candidate_json output/disease2drug/${Disease}.disease2drug.candidate.jsonl \
  --output output/LINCS/${Disease}/LINCS.search.${Disease}.rank.tsv \
  --detailed_output output/LINCS/${Disease}/LINCS.search.${Disease}.rank.detailed.tsv \
  --top_n 200 \
  --dose_weight 0.2 \
  --ic50_weight 0.8 \
  --dose_k 0.5


########################################
# 5.3 GSEA enrichment analysis
#     - Identify biological processes & pathways
#     - GO BP + KEGG
########################################
python 4.3.LINCS.enrich.py \
  --input output/LINCS/${Disease}/LINCS.search.${Disease}.rank.tsv \
  --species Human \
  --dbs GO_Biological_Process_2021 KEGG_2021_Human \
  --p_cutoff 0.05 \
  --outdir output/LINCS_enrich/${Disease}


########################################
# 5.4 GPT-based GSEA summarization
#     - Convert enrichment results into
#       disease-relevant mechanistic narratives
########################################
python 4.4.LINCS.GPT.py \
  --disease ${Disease} \
  --rank_tsv output/LINCS/${Disease}/LINCS.search.${Disease}.rank.tsv \
  --gsea_dir output/LINCS_enrich/${Disease} \
  --prompt prompts/4.4.GSEA_Summarizer.txt \
  --param_file parameter.gpt4o.txt \
  --output output/LINCS_enrich_summary/${Disease}.GSEA.summary.jsonl


########################################
# 5.5 Mechanistic drug prediction
#     - Infer drug mechanisms from GSEA summaries
########################################
python 4.5.RepurposeAgent.disease2drug.prediction.mechanistic.py \
  --summary_file output/LINCS_enrich_summary/${Disease}.GSEA.summary.jsonl \
  --prompt_file prompts/4.5.GSEA_Summarizer.mechanistic.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/LINCS_enrich_summary_mechanistic/${Disease}


###############################################################################
# 6. Evidence fusion & disease subtype modeling
###############################################################################

########################################
# 6.1 Final evidence aggregation
#     - Combine KG, literature, LINCS, GSEA
########################################
python 5.RepurposeAgent.disease2drug.Summarization.py \
  --disease2drug_jsonl output/disease2drug/${Disease}.disease2drug.candidate.jsonl \
  --drug2gene_dir output/drug2gene/${Disease} \
  --gsea_dir output/LINCS_enrich_summary_mechanistic/${Disease} \
  --out_dir output/final_input/${Disease}


########################################
# 6.2 Disease subtype generation
#     - LLM-based disease stratification
########################################
python 6.0.RepurposeAgent.disease2drug.subtype_generation.py \
  --disease input/${Disease}.json \
  --prompt prompts/6.0.SubtypeGeneration.txt \
  --list_prompt prompts/6.0.SubtypelistGeneration.txt \
  --param_file parameter.gpt4o.txt \
  --output prompts/subtype/${Disease}.subtype.statement.txt \
  --output_list prompts/subtype/${Disease}.subtype.txt


########################################
# 6.3 Subtype-aware drug prediction
#     - Rank drugs per disease subtype
#
# option:
# --prompt_file prompts/6.1.Final.rule.txt for using heuristic rule for scoring.
# --prompt_file prompts/6.1.Final.rule.txt for using heuristic rule for scoring.
#
########################################
python 6.1.RepurposeAgent.disease2drug.prediction.py \
  --input_dir output/final_input/${Disease} \
  --prompt_file prompts/6.1.Final.txt \
  --param_file parameter.gpt4o.txt \
  --out_dir output/final_prediction/${Disease} \
  --subtypes_file prompts/subtype/${Disease}.subtype.txt \
  --subtype_statements_file prompts/subtype/${Disease}.subtype.statement.txt \
  --json_input input/${Disease}.json


###############################################################################
# 7. Final output formatting
#    - Convert JSON predictions into TSV
###############################################################################
python 7.RepurposeAgent.disease2drug.json2tsv.py \
  --input_folder output/final_prediction/${Disease} \
  --output_tsv output/${Disease}.final_prediction.tsv


###############################################################################
# 8. Categorization
###############################################################################
python 8.RepurposeAgent.categorization.py \
  --input  output/${Disease}.final_prediction.tsv \
  --output output/${Disease}.final_prediction.Categorization.tsv \
  --prompt prompts/8.Categorization.txt

python 9.RepurposeAgent.filtering.py \
  --input  output/${Disease}.final_prediction.Categorization.tsv \
  --output output/${Disease}.final_prediction.Categorization.filtered.tsv


echo "RepurposeAgent pipeline completed for disease: ${Disease}"
###############################################################################
