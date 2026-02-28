# 🏠 Home Depot Product Search Relevance
Predicting product search relevance for Home Depot using TF-IDF + Linear Regression and Word2Vec + PyTorch 

> Based on the [Home Depot Product Search Relevance](https://www.kaggle.com/competitions/home-depot-product-search-relevance) Kaggle competition.

---

## Problem Statement

Given a search query (e.g. *"angle bracket"*) and a product title + description, predict a **relevance score between 1 and 3**:
- `1` → Not relevant
- `2` → Partially relevant
- `3` → Highly relevant

This is a **regression problem** evaluated on RMSE.

---

## What's in the Notebook

1. **Data Loading** — merged train, test, product descriptions, and attributes CSVs
2. **EDA** — distribution of relevance scores, relevance vs. search term length, scatter plots
3. **Feature Engineering** — combined search term + product title + description into a single text field
4. **ML Approach** — TF-IDF vectorization → Linear Regression
5. **Deep Learning Approach** — Word2Vec embeddings → PyTorch neural network (SimpleNN)
6. **Submission** — predictions saved to `submission.csv`

---

## Two Approaches at a Glance

| Approach | Text Representation | Model |
|----------|-------------------|-------|
| Machine Learning | TF-IDF | Linear Regression |
| Deep Learning | Word2Vec (100d) | PyTorch SimpleNN (MLP) |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERT_Tokenizer-yellow?logo=huggingface)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/home-depot-relevance.git
cd home-depot-relevance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
kaggle competitions download -c home-depot-product-search-relevance
```
Or download manually from [Kaggle](https://www.kaggle.com/competitions/home-depot-product-search-relevance/data) and place the files in a `data/` folder.

### 4. Run the notebook
```bash
jupyter notebook home-depot-product-relevance-challenge.ipynb
```

---

## Project Structure

```
home-depot-relevance/
├── home-depot-product-relevance-challenge.ipynb   # Main notebook
├── requirements.txt                                # Dependencies
├── .gitignore                                      # Files to exclude
└── README.md
```

> ⚠️ Dataset files are not included due to Kaggle's terms of use. Download them directly from the competition page.

---

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Linear Regression (TF-IDF) | MSE / MAE | See notebook |
| SimpleNN (Word2Vec) | MSE | See notebook |

---

## License

MIT
