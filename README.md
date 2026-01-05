# KG×LM  
## Knowledge Graphs Meet Large Language Models: A New Paradigm for Drug Repurposing

**Chih-Hsuan Wei¹\***, **Zhizheng Wang¹\***, **Chi-Ping Day²\***, Joey Chan¹˒³, Betty Tyler⁴, Hasan Slika⁴, Kyle Turner⁴,  
Christine C. Alewine⁵, Chin-Hsien Tai², Shubo Tian¹, Chi-Ping’friend⁶, and **Zhiyong Lu¹†**

\* These authors contributed equally  
† Corresponding author

---

## Overview

**KG×LM** is a disease-centric framework that integrates **biomedical knowledge graphs (KGs)** with **large language models (LLMs)** to enable **systematic drug repurposing and disease-specific drug ranking**.

The framework combines:
- Structured knowledge (drug–gene–disease relations)
- Unstructured evidence (biomedical literature)
- Perturbational transcriptomics (LINCS)
- LLM-based reasoning and summarization

Unlike conventional drug-repurposing pipelines that focus on binary drug–disease prediction, KG×LM provides an **interpretable, multi-evidence ranking of candidate drugs**, with optional **mechanistic explanations and disease subtype–aware analysis**.

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
Disease Normalization & Attribute Extraction
   ↓
Disease → Drug / Gene Candidate Retrieval (KG)
   ↓
Drug–Gene Evidence Integration (CTD, PubTator, DGIdb)
   ↓
LINCS Perturbation & Drug Ranking
   ↓
GSEA Pathway Enrichment
   ↓
LLM-based Mechanistic Summarization
   ↓
Subtype-aware Drug Ranking
```

---

## Repository Structure

```
KGxLM/
├── AutoRun.sh                     # End-to-end pipeline
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
- CUDA (optional, for acceleration)

### Setup

```bash
git clone https://github.com/your-org/KGxLM.git
cd KGxLM

conda create -n kgxlm python=3.11
conda activate kgxlm

pip install -r requirements.txt
```

---

## LLM Configuration

KG×LM uses OpenAI-compatible LLMs.

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
bash AutoRun.sh "Acute Myeloid Leukemia"
```

---

## Output

Main output file:

```
output/<Disease>.final_prediction.tsv
```

Each result includes:
- Drug name
- Aggregated ranking score
- Supporting genes
- Literature evidence
- LINCS perturbation support
- Mechanistic interpretation

---

## Reproducibility

For full environment capture:

```bash
conda env export > environment.yml
```

---

## Citation

If you use **KG×LM**, please cite:

```bibtex
@article{KGxLM,
  title={Knowledge Graphs Meet Large Language Models: A New Paradigm for Drug Repurposing},
  author={Wei, Chih-Hsuan and Wang, Zhizheng and Day, Chi-Ping and others},
  journal={TBD},
  year={2025}
}
```

---

## Disclaimer

KG×LM is intended for **research and hypothesis generation only** and is not designed for direct clinical decision-making.

---

## Contact

For questions or collaborations, please open a GitHub issue or contact the corresponding author.
