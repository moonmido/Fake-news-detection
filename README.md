# 🕵️ Fake News Detection

A machine learning project that classifies news articles as **real** or **fake** using Natural Language Processing (NLP) and Logistic Regression.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Example Predictions](#example-predictions)
- [Technologies Used](#technologies-used)
- [Results](#results)

---

## Overview

This project builds a simple yet effective fake news detector. It takes a news article's text as input and predicts whether the article is real or fake. The pipeline uses TF-IDF vectorization to convert raw text into numerical features, then trains a Logistic Regression classifier on labeled news data.

---

## Project Structure

```
Fake-news-detection/
│
├── fake_news_detection.py          # Main Python script
├── Fake_news_detection (1).ipynb   # Jupyter Notebook version
├── evaluation.csv                  # Dataset (text + label columns)
└── README.md
```

---

## Dataset

The project expects a CSV file named `evaluation.csv` with a semicolon (`;`) delimiter containing at least these two columns:

| Column  | Description                          |
|---------|--------------------------------------|
| `text`  | The body/content of the news article |
| `label` | The ground truth label (real / fake) |

> **Note:** Rows with missing values are automatically dropped before training.

---

## How It Works

1. **Load Data** — Reads `evaluation.csv`, drops null rows.
2. **Vectorize Text** — Applies `TfidfVectorizer` with English stop words removed and a max document frequency of 0.7 to filter overly common terms.
3. **Train/Test Split** — Splits the data 80/20 (training/testing) with a fixed random seed for reproducibility.
4. **Train Model** — Fits a `LogisticRegression` classifier on the training set.
5. **Evaluate** — Prints accuracy on the test set.
6. **Predict** — Exposes a `predict_news(text)` function for classifying new articles.

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install pandas scikit-learn
```

---

## Usage

### Run the Python Script

```bash
python fake_news_detection.py
```

### Run the Jupyter Notebook

```bash
jupyter notebook "Fake_news_detection (1).ipynb"
```

### Use the Prediction Function

```python
from fake_news_detection import predict_news

result = predict_news("Your news article text goes here")
print(result)  # e.g., ['REAL'] or ['FAKE']
```

---

## Example Predictions

```python
predict_news("Government announces new education reform")
# → ['REAL']

predict_news("Aliens landed in Algeria yesterday and took control")
# → ['FAKE']
```

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading and preprocessing |
| `scikit-learn` | TF-IDF vectorization, model training, evaluation |
| `TfidfVectorizer` | Convert text to numerical features |
| `LogisticRegression` | Binary classification model |
| `Jupyter Notebook` | Interactive development and exploration |

---

## Results

After training, the model prints the accuracy score on the held-out test set:

```
Accuracy: 95.5
```

> Actual accuracy depends on the dataset used.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## License

This project is open source. Feel free to use and adapt it for your own work.
