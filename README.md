# NLP Predictive Typing System

A comprehensive exploration and comparison of language models for next-word prediction in predictive typing systems. This project evaluates statistical N-gram models, advanced KenLM implementations, and neural architectures (LSTM, GPT-2) on a large-scale corpus.

**Final Result: 63.60% accuracy** using 5-gram KenLM with Modified Kneser-Ney smoothing.

## Table of Contents
- [Overview](#overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)

## Overview

This project addresses the "Predictive Typing Contest 2" challenge: building a language model that predicts the next word in a sequence given previous context and the first letter of the target word. This simulates predictive keyboard functionality on mobile devices.

The implementation explores a progression from traditional statistical methods to state-of-the-art neural architectures, providing a comprehensive performance comparison across different approaches.

## Task Description

**Goal**: Predict the next word given:
- Previous context (preceding words)
- First letter of the target word (constraint)

**Constraints**:
- No spell-checking allowed
- Must handle punctuation marks, numbers, and symbols (~14.3% of targets)
- Evaluation based on exact match accuracy

## Dataset

### Training Set
- **Size**: 3,803,957 sentences (128M tokens)
- **Vocabulary**: 99,021 unique tokens
- **Quality**: High density - only 8.31% of words appear ≤5 times
- **Tokenization**: Penn Treebank style (contractions split: "it's" → "it", "'", "s")
- **Domain**: News/financial reports (evident from frequent numbers and weekdays)

### Development Set
- **Size**: 94,825 instances
- **Context Length**: Mean 24.95 words, 95th percentile at 37 words
- **First Letter Distribution**:
  - Most common: 't' (13.64%), 'a' (10.46%), 's' (7.12%)
  - Non-alphabetic targets: 14.3% (comma, hyphen, numbers)

### Training Strategy
Progressive sampling approach with four dataset sizes:
- **10K sentences**: Quick prototyping and syntax validation
- **100K sentences**: Early performance assessment
- **1M sentences**: Scaled training for promising models
- **3.8M sentences (Full)**: Final training for best-performing models

**Scaling Criteria**:
- Performance improvement: ≥5 percentage points gain
- Training time: <2 hours per epoch for neural models

## Models Implemented

### 1. Statistical N-gram Models

Custom implementations with backoff strategy:

| Model | Best Accuracy | Training Size |
|-------|---------------|---------------|
| Trigram (3-gram) | 58.12% | Full (3.8M) |
| 4-gram | 55.23% | 1M |
| 5-gram | 55.67% | 1M |
| 6-gram | 50.68% | 100K |

**Features**:
- Sentence boundary markers (`<s>`, `</s>`)
- Recursive backoff strategy (n-gram → (n-1)-gram → ... → unigram)
- First-letter fallback: Most frequent word matching the required letter

**Notebooks**:
- [2_Trigram_Model.ipynb](2_Trigram_Model.ipynb)
- [3_Fourgram_Model.ipynb](3_Fourgram_Model.ipynb)
- [4_Fivegram_Model.ipynb](4_Fivegram_Model.ipynb)
- [10_Sixgram_Model.ipynb](10_Sixgram_Model.ipynb)

### 2. KenLM with Kneser-Ney Smoothing

**Best Model**: 5-gram KenLM - **63.60% accuracy** (Full dataset)

Advanced statistical model using Modified Kneser-Ney smoothing for superior handling of unseen n-grams.

**Performance Scaling**:
- 10K sentences: 46.21%
- 100K sentences: 52.80%
- 1M sentences: 57.30%
- 3.8M sentences: **63.60%**

**Notebook**: [5_KenLM.ipynb](5_KenLM.ipynb)

### 3. LSTM (Long Short-Term Memory)

Custom RNN architecture with novel first-letter constraint integration.

**Architecture**:
- Multi-layer LSTM for context encoding
- Separate embedding layer for first-letter constraint
- Concatenated features → linear layer → vocabulary distribution

**Hyperparameters** (tuned via grid search):
- Embedding dimension: **256**
- Hidden dimension: **512**
- Number of layers: **2**
- Dropout probability: **0.4**
- Optimizer: Adam

**Performance**: 40.23% (100K sentences) - underfit due to data/time constraints

**Notebooks**: [LSTM_10000.ipynb](LSTM_10000.ipynb), [LSTM_100000.ipynb](LSTM_100000.ipynb)

### 4. GPT-2 (Zero-Shot)

Pre-trained transformer model evaluated without fine-tuning.

**Approach**:
- Constrained decoding strategy
- Top-k sampling (k=1000) with first-letter filtering
- No training on contest dataset

**Performance**: 55.95% (zero-shot) - strong generalization but outperformed by in-domain statistical models

**Notebook**: [12_GPT2_Model.ipynb](12_GPT2_Model.ipynb)

### 5. Additional Experiments

- [6_Model_Ensemble.ipynb](6_Model_Ensemble.ipynb) - Ensemble methods
- [7_Neural_Cache_LSTM.ipynb](7_Neural_Cache_LSTM.ipynb) - LSTM with caching
- [8_BiLSTM_Ngram.ipynb](8_BiLSTM_Ngram.ipynb) - Bidirectional LSTM with n-gram features
- [9_Ngram_LSTM_Interpolation.ipynb](9_Ngram_LSTM_Interpolation.ipynb) - Hybrid approach
- [11_Sevengram_Model.ipynb](11_Sevengram_Model.ipynb) - 7-gram model
- [13_Transformer_Scratch.ipynb](13_Transformer_Scratch.ipynb) - Custom transformer

## Results

Detailed experimental results and analysis are available in:
- [NLP__Contest_2-1047951-17608926027974.pdf](NLP__Contest_2-1047951-17608926027974.pdf)

### Overall Performance Comparison

| Training Size | 3-gram | 4-gram | 5-gram | 6-gram | KenLM (5-gram) | LSTM | GPT-2 |
|---------------|--------|--------|--------|--------|----------------|------|-------|
| 10K | 43.42% | 43.54% | 43.79% | 43.78% | 46.21% | 31.47% | - |
| 100K | 49.89% | 50.41% | 50.73% | 50.68% | 52.80% | 40.23% | - |
| 1M | 54.08% | 55.23% | 55.67% | - | 57.30% | - | - |
| **Full (3.8M)** | **58.12%** | - | - | - | **63.60%** | - | - |
| Pre-trained | - | - | - | - | - | - | 55.95% |

## Key Findings

### 1. Statistical Models Outperform Neural Approaches

The **5-gram KenLM model achieved 63.60%**, outperforming:
- Custom LSTM (40.23%)
- Zero-shot GPT-2 (55.95%)
- Simple n-grams (58.12% max)

**Why?**
- Superior smoothing (Modified Kneser-Ney) handles data sparsity effectively
- Computational efficiency enabled training on full dataset
- Task-specific optimization vs. general-purpose neural models

### 2. Data Scaling is Critical

Every model showed consistent improvement with more data:

**KenLM Scaling**:
- 10K → 100K: +6.59 percentage points
- 100K → 1M: +4.50 percentage points
- 1M → 3.8M: +6.30 percentage points

**Conclusion**: Data quantity is the single most important factor for language model performance.

### 3. Simple Models + More Data > Complex Models + Less Data

The simple **3-gram model (full data): 58.12%** outperformed the **5-gram model (1M data): 55.67%**

This demonstrates the "brute force" approach effectiveness when computational resources allow.

### 4. Neural Models Require Massive Scale

- **LSTM underfitting**: 40.23% despite sophisticated architecture
- **GPT-2 generalization**: 55.95% without any in-domain training shows power of pre-training
- Both require significantly more data and compute than available in this project

### 5. Context Window Trade-offs

Higher-order n-grams showed diminishing returns:
- 3-gram → 4-gram: Significant improvement
- 5-gram → 6-gram: Performance degradation (data sparsity)
- **Optimal**: 5-gram with proper smoothing (KenLM)

## Evaluation Metrics

**Primary Metric**: Exact Match Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

A prediction is correct only if it **exactly matches** the target word (case-sensitive).

**Development Set**: 94,825 instances

