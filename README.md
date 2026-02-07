## DrugKLM: Knowledge Graphs Meet Large Language Models: A New Paradigm for Drug Repurposing

## Overview

**DrugKLM** is a disease-centric framework that integrates **biomedical knowledge graphs (KGs)** with **large language models (LLMs)** to enable **systematic drug repurposing and disease-specific drug ranking**.

The framework combines:
- Structured knowledge (drug–gene–disease relations)
- Unstructured evidence (biomedical literature)
- Perturbational transcriptomics (LINCS)
- LLM-based reasoning and summarization

Unlike conventional drug-repurposing pipelines that focus on binary drug–disease prediction, DrugKLM provides an **interpretable, multi-evidence ranking of candidate drugs**, with optional **mechanistic explanations and disease subtype–aware analysis**.

---

## Key Contributions

- Unified KG + LLM framework for drug repurposing
- Disease-specific drug ranking instead of binary association prediction
- Multi-source evidence integration (KG, literature, LINCS, GSEA)
- Mechanistic hypothesis generation grounded in biological pathways
- End-to-end automated pipeline

---

## System Pipeline

```
Disease Query
   ↓
Disease Standardization
   ↓
Candidate Drug Generation and Evidence Integration
   ↓
Disease- and Drug-related Gene Evidence Integration
   ↓
Pathway Perturbation Analysis (LINCS / GSEA)
   ↓
Evidence-grounded Confidence Scoring and Drug Ranking

```

---

## Repository Structure

```
DrugKLM/
├── DrugKLM.sh                     # End-to-end pipeline
├── input/                         # Disease input files
├── output/                        # Generated results
├── prompts/                       # LLM prompts
├── DB/                            # Knowledge bases and omics data
├── MedCPT.npy/                    # Disease embeddings
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python ≥ 3.11
- Conda (recommended)
- Java (required for GSEA)
- Download **[files](https://ftp.ncbi.nlm.nih.gov/pub/lu/DrugKLM/)** and store in DrugKLM folder
### Setup

```bash
git clone https://github.com/your-org/DrugKLM.git
cd DrugKLM

conda create -n DrugKLM python=3.11
conda activate DrugKLM

pip install -r requirements.txt
```

---

## LLM Configuration

DrugKLM uses OpenAI-compatible LLMs.

Create a parameter file (not committed to GitHub):

```text
parameter.gpt4o.txt
```

Example:
```text
OPENAI_API_KEY=your_api_key
MODEL=gpt-4o
```

---

## Usage

Run the complete pipeline for a disease:

```bash
bash DrugKLM.sh "Acute Myeloid Leukemia"
```

---

## Output

Main output file:

```
output/<Disease>.final_prediction.tsv
```

---

## Citation

If you use **DrugKLM**, please cite:

```bibtex
@article{DrugKLM,
  title={DrugKLM: Knowledge Graphs Meet Large Language Models: A New Paradigm for Drug Repurposing},
  author={Wei, Chih-Hsuan and Wang, Zhizheng and Day, Chi-Ping and others},
  journal={TBD},
  year={2026}
}
```

---

## Disclaimer

DrugKLM is intended for **research and hypothesis generation only** and is not designed for direct clinical decision-making. DrugKLM shows the results of research conducted in the Computational Biology Branch, DIR/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. 

---

## Contact

For questions or collaborations, please open a GitHub issue or contact the author, Chih-Hsuan Wei (chih-hsuan.wei@nih.gov).
