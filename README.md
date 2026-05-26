# 📰 Multilabel News Article Classification & Entity-Aware Summarisation Engine

## 🎯 Project Overview
A complete NLP pipeline that automatically classifies news articles into multiple topic categories and generates concise summaries. Built for media monitoring platforms, brand reputation tracking, and financial news intelligence.

## 📊 Dataset
- **Train set:** 3000 articles
- **Test set:** 498 articles
- **Labels:** 10 topic categories (Politics, Economy, Health, Technology, Sports, Crime, Entertainment, Environment, Science, International)
- **Source domains:** BBC, Reuters, Al Jazeera and 19 others

## 🔧 Tech Stack
- **Language:** Python
- **Platform:** Google Colab (T4 GPU)
- **Libraries:** PyTorch, HuggingFace Transformers, spaCy, scikit-learn, VADER, ROUGE, BERTScore

## 🚀 Pipeline Steps

### 1. Text Preprocessing
- HTML tag removal using BeautifulSoup
- Unicode normalisation
- Language detection using langdetect
- Filtered non-English articles programmatically
- Lowercasing and whitespace normalisation

### 2. Exploratory Data Analysis (EDA)
- Headline and body text length distributions
- Missing value analysis
- Label distribution across 10 categories

### 3. Baseline Models
| Model | Micro F1 | Macro F1 | Hamming Loss | Jaccard |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.7011 | 0.6939 | 0.1546 | 0.5548 |
| TF-IDF + Linear SVM | 0.5006 | 0.5873 | 0.4724 | 0.3307 |
| Word2Vec + MLP | 0.4108 | 0.4079 | 0.6730 | 0.2592 |

### 4. RoBERTa Fine-tuning (Multilabel Classification)
- Fine-tuned `roberta-base` with sigmoid output head
- BCEWithLogitsLoss for multilabel classification
- Per-label threshold tuning (not global 0.5)
- Trained for 25 epochs on T4 GPU

**Final Test Prediction Results:**
| Metric | Score |
|---|---|
| Micro F1 | **0.9019** |
| Macro F1 | **0.9062** |
| Hamming Loss | 0.0500 |
| Jaccard Score | 0.8432 |

### 5. Named Entity Recognition (NER)
- Used spaCy `en_core_web_trf` model
- Extracted 41,269 entities from 2838 train articles
- Average 14.54 entities per article
- Top entity types: ORG, GPE, DATE, CARDINAL, PERSON

### 6. Abstractive Summarisation
- Used pretrained `t5-small` for inference
- Entity grounding constraint applied
- Evaluated on full 498 test articles

| Metric | Score |
|---|---|
| ROUGE-1 | 0.6538 |
| ROUGE-2 | 0.5856 |
| ROUGE-L | 0.6379 |
| BERTScore F1 | **0.9292** |
| Entity Grounding | **98.80%** |

### 7. Misinformation Signal Scoring
Five rule-based signals engineered:
- Clickbait score (VADER sentiment intensity)
- Emotional language ratio (NRC lexicon)
- Source credibility (domain whitelist)
- Factual density (named entities per 100 words)
- Quote authenticity (direct vs indirect quotes)
- Combined into weighted Mis-Risk Score [0-1]

## 📈 Key Results Summary
| Component | Metric | Score |
|---|---|---|
| RoBERTa Classification | Micro F1 | **0.9019** |
| T5 Summarisation | BERTScore F1 | **0.9292** |
| NER | Entity Grounding | **98.80%** |
| Baseline (TF-IDF+LR) | Micro F1 | 0.7011 |

## ⚠️ Important Notes
- `summary_ref`, `entities_ref`, `mis_risk_label` never used as input features (data leakage prevention)
- Per-label threshold tuning applied for all classifiers
- Language filtering done programmatically via langdetect
- GPU: Google Colab T4

## 👤 Author
Esthaking
