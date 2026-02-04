# LLM Financial Sentiment Analysis

A comprehensive experiment comparing traditional machine learning baselines with fine-tuned Large Language Models (RoBERTa variants) for financial text sentiment classification.

## Project Overview

This project evaluates various approaches to sentiment analysis on financial text, progressing from traditional ML baselines to state-of-the-art transformer models with parameter-efficient fine-tuning techniques.

**Best Result:** 87.67% accuracy using RoBERTa-Large with LoRA and Weighted Focal Loss

## Dataset

**Source:** Kaggle Financial Sentiment Analysis dataset (combined FiQA + Financial PhraseBank)

| Metric | Value |
|--------|-------|
| Total samples | 5,842 |
| Classes | 3 (Negative, Neutral, Positive) |
| Train/Test split | 80/20 |

**Class Distribution:**
- Neutral: 3,130 (53.6%)
- Positive: 1,852 (31.7%)
- Negative: 860 (14.7%)

## Project Structure

```
llm_financial_sentiment/
├── data/
│   └── data.csv                    # Combined dataset
├── notebooks/
│   ├── 00_download_data.ipynb      # Data acquisition
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_ml_baselines.ipynb       # Traditional ML models
│   ├── 03_roberta_base.ipynb       # RoBERTa-base full fine-tune
│   ├── 04_roberta_large.ipynb      # RoBERTa-large full fine-tune
│   ├── 05_roberta_large_lora.ipynb # RoBERTa-large with LoRA
│   ├── 06_error_analysis.ipynb     # Error analysis (LoRA model)
│   ├── 07_roberta_large_lora_weighted_focal_loss.ipynb
│   ├── 08_error_analysis_weighted_focal_loss.ipynb
│   ├── 09a_calibration_full_finetune.ipynb
│   ├── 09b_calibration_weighted_focal_loss.ipynb
│   └── utils.py                    # Text preprocessing utilities
└── results/
    ├── checkpoints/                # Trained model weights
    ├── predictions/                # Model predictions on test set
    └── calibration/                # Calibration analysis results
```

## Notebooks Summary

### Data Preparation & Exploration

| Notebook | Purpose |
|----------|---------|
| `00_download_data` | Downloads dataset from Kaggle |
| `01_eda` | Exploratory analysis: class distribution, sentence lengths, word clouds |

### Model Training

| Notebook | Model | Accuracy | Key Details |
|----------|-------|----------|-------------|
| `02_ml_baselines` | Traditional ML (9 models) | 73.4% (SVM) | TF-IDF features |
| `03_roberta_base` | RoBERTa-base (110M params) | 85.35% | Full fine-tune |
| `04_roberta_large` | RoBERTa-large (355M params) | 86.01% | Full fine-tune |
| `05_roberta_large_lora` | RoBERTa-large + LoRA | 87.04% | 2.24% params trained |
| `07_roberta_large_lora_weighted_focal_loss` | RoBERTa-large + LoRA + Focal Loss | **87.67%** | Best model |

### Analysis

| Notebook | Purpose |
|----------|---------|
| `06_error_analysis` | Confusion matrices, misclassification patterns (LoRA model) |
| `08_error_analysis_weighted_focal_loss` | Error analysis for focal loss model |
| `09a_calibration_full_finetune` | Calibration metrics, temperature scaling |
| `09b_calibration_weighted_focal_loss` | Calibration analysis for best model |

## Results Summary

### Model Performance Comparison

| Model | Accuracy | Macro F1 | Macro AUROC | GPU Memory |
|-------|----------|----------|-------------|------------|
| ML Baseline (SVM) | 73.4% | - | - | CPU |
| RoBERTa-Base | 85.35% | - | - | 4.5 GB |
| RoBERTa-Large (Full FT) | 86.01% | 0.8151 | 0.9676 | 11.3 GB |
| RoBERTa-Large (LoRA) | 87.04% | 0.8194 | 0.9715 | 10.3 GB |
| **RoBERTa-Large (LoRA + Focal)** | **87.67%** | - | - | - |

### Calibration Results

| Model | ECE | MCE | Optimal Temperature |
|-------|-----|-----|---------------------|
| RoBERTa-Large (Full FT) | 0.0514 | 0.2517 | 1.467 |
| RoBERTa-Large (LoRA + Focal) | 0.0728 | 0.1475 | 0.644 |

## Key Findings

1. **Transformer Superiority:** Fine-tuned RoBERTa models achieve ~14% improvement over best traditional ML baseline.

2. **LoRA Efficiency:** Parameter-efficient fine-tuning with LoRA achieves superior results while training only 2.24% of model parameters, reducing memory usage and checkpoint sizes.

3. **Class Imbalance Mitigation:** Weighted focal loss effectively addresses the imbalanced dataset (14.7% negative class), yielding the best overall performance.

4. **Calibration Matters:** Temperature scaling improves model confidence estimates. The focal loss model shows better calibration properties (lower MCE).

5. **Neutral Class Challenge:** The majority neutral class (53.6%) tends to act as a "trash can" for uncertain predictions, a common pattern in imbalanced sentiment classification.

## Technical Stack

- **Deep Learning:** PyTorch, HuggingFace Transformers
- **Efficient Fine-tuning:** PEFT (LoRA)
- **ML Baselines:** scikit-learn, XGBoost, LightGBM
- **Visualization:** matplotlib, seaborn, WordCloud

## Text Preprocessing

The `utils.py` module provides preprocessing functions:
- URL and email removal
- Contraction expansion
- Lowercasing and non-ASCII removal
- Stemming (Snowball) and lemmatization (WordNet)
- Custom stopword removal

## Usage

1. **Setup:** Run `00_download_data.ipynb` to acquire the dataset
2. **Explore:** Review `01_eda.ipynb` for data insights
3. **Train:** Execute training notebooks (02-07) sequentially
4. **Analyze:** Review error analysis and calibration notebooks

## Saved Artifacts

- **Checkpoints:** 4 fine-tuned model weights in `results/checkpoints/`
- **Predictions:** Test set predictions with probabilities in `results/predictions/`
- **Calibration:** Temperature scaling parameters in `results/calibration/`
