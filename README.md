---
title: PubTator3 Tagger (Offline Package)
license: apache-2.0
language:
  - en
tags:
  - biomedical
  - pubmed
  - pubtator
  - named-entity-recognition
  - text-mining
pretty_name: PubTator3 Offline Tagger
size_categories:
  - huge
---

# PubTator3 Tagger (Offline Package)
This dataset provides the complete **PubTator3 offline tagger package**, distributed as a multi-part compressed archive for large-scale biomedical text mining.

The package supports batch annotation of PubMed / PMC-scale corpora and is intended for **research use only**.

---

## Package Distribution

The tagger is distributed as multiple compressed files:

PubTator3_tagger.tar.gz.part_*

All parts must be downloaded and placed in the same directory before reconstruction.

---

## Installation

### 1. Reconstruct the archive

bash
cat PubTator3_tagger.tar.gz.part_* > PubTator3_tagger.tar.gz

### 2. Extract the package

bash
tar -xzf PubTator3_tagger.tar.gz

After extraction, a directory named 'PubTator3_tagger/' will be created, containing all required files and scripts.

### 3. Install required dependencies

bash
./Installation.sh

---

## ▶️ Usage

Detailed usage instructions are provided inside the extracted 'PubTator3_tagger/' directory.

### Input locations

* Free-text input:
  
  ${DIR}/Data/input_txt

* BioC XML format:
  
  ${DIR}/Data/input_bioc

### Run the batch tagging pipeline

bash
./PubTator.sh

---

## Intended Use
These tools are intended for **research use in biomedical text mining and natural language processing**, including large-scale literature annotation and information extraction.

This repository **does not provide an online inference API**.

---
## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, DIR/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. 
---

## Citation
Please cite the relevant publications below.

1. Wei C-H, Allot A, Lai P-T, Leaman R, Tian S, Luo L, Jin Q, Wang Z, Chen Q, Lu Z. PubTator 3.0: an AI-powered literature resource for unlocking biomedical knowledge. Nucleic Acids Research. 2024;51(W1):W540–W546.

## Taggers
Taggers of PubTator3 are listed below.

1. Wei C-H, Luo L, Islamaj R, Lai P-T, Lu Z. GNorm2: an improved gene name recognition and normalization system. Bioinformatics. 2023;39(10):btad599.
2. Wei C-H, Allot A, Riehle K, Milosavljevic A, Lu Z. tmVar 3.0: an improved variant concept recognition and normalization tool. Bioinformatics. 2022;38(18):4449–4451.
3. Lai P-T, Wei C-H, Tian S, Leaman R, Lu Z. Enhancing Biomedical Relation Extraction with Directionality. Bioinformatics. 2025;41(Supplement_1):i68–i76.
4. Islamaj R, Leaman R, Kim S, Kwon D, Wei C-H, Comeau DC, Peng Y, Cissel D, Coss C, Fisher C, Guzman R, Kochar PG, Koppel S, Trinh D, Sekiya K, Ward J, Whitman D, Schmidt S, Lu Z. NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature. Scientific Data. 2021;8(1):91.
5. Leaman R, Lu Z. TaggerOne: joint named entity recognition and normalization with semi-Markov Models. Bioinformatics. 2016;32(18):2839–2846.
