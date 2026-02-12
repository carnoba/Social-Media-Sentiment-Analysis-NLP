# 📊 Twitter and Reddit Sentiment Analysis Dataset

## 🚀 Project Overview

This project is a comprehensive **Sentiment Analysis Pipeline** designed to process, analyze, and visualize public sentiment from **Twitter and Reddit** datasets. It leverages Natural Language Processing (NLP) techniques and Machine Learning models to classify social media comments into **Positive**, **Neutral**, or **Negative** sentiments.

The system is built to check for existing processed data to optimized performance, compares **Traditional Machine Learning** (Logistic Regression) against **Deep Learning** approaches, and generates actionable visualizations for brand or topic analysis.

---

## 🔑 Key Features

- **📂 Data Ingestion & Merging:** Seamlessly combines datasets from Twitter and Reddit.
- **🧹 Advanced Preprocessing:**
  - Automated cleaning (URL/HTML tag removal, special character filtering).
  - Tokenization & Lemmatization using `NLTK`.
  - Intelligent sampling to handle large datasets efficiently.
- **🏷️ Automated Labeling:** Utilizes `TextBlob` to generate/verify sentiment labels.
- **🤖 Model Comparison:**
  - **Logistic Regression (TF-IDF):** A strong baseline for text classification.
  - **Deep Learning (Keras/TensorFlow):** A dense neural network for capturing complex patterns.
  - **Performance Metrics:** Precision, Recall, F1-Score, and Accuracy comparisons.
- **📈 Insightful Visualizations:**
  - **Word Clouds:** Visual representation of common positive and negative terms.
  - **Sentiment Distribution:** Breakdown of sentiment classes.
  - **Trend Analysis:** Simulated time-series data to visualize sentiment shifts over time.

---

## 🛠️ Technology Stack

- **Language:** Python 3.8+
- **Data Manipulation:** `Pandas`, `NumPy`
- **NLP & Text Processing:** `NLTK`, `TextBlob`, `Regular Expressions (re)`
- **Machine Learning:** `Scikit-learn` (Logistic Regression, TF-IDF)
- **Deep Learning:** `TensorFlow`, `Keras`
- **Visualization:** `Matplotlib`, `Seaborn`, `WordCloud`

---

## 📂 Project Structure

```bash
📦 Twitter-Reddit-Sentiment-Analysis
├── 📄 main.py               # 🚀 Entry point: Orchestrates loading, modeling, and visualization
├── 📄 preprocessing.py      # 🧹 Data cleaning, tokenization, lemmatization, and sampling
├── 📄 models.py             # 🤖 Model training (LogReg vs DL) and evaluation
├── 📄 visualization.py      # 📊 Generates charts and word clouds
├── 📄 check_libs.py         # 🛠️ Utility to check installed libraries
├── 📄 report_structure.md   # 📝 Outline for the final analysis report
├── 💾 Reddit_Data.csv       # 📂 Raw Reddit Dataset
├── 💾 Twitter_Data.csv      # 📂 Raw Twitter Dataset
├── 💾 processed_data.csv    # ⚡ Cache of cleaned data (generated after first run)
└── 🖼️ *.png                 # 📊 Generated images (WordClouds, distributions)
```

---

## ⚙️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/twitter-reddit-sentiment.git
   cd twitter-reddit-sentiment
   ```

2. **Install Dependencies**
   Ensure you have Python installed. It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

   _If `requirements.txt` is missing, manually install:_

   ```bash
   pip install pandas numpy scikit-learn nltk textblob tensorflow matplotlib seaborn wordcloud
   ```

3. **Download NLTK Data**
   The script automatically downloads necessary NLTK data (`punkt`, `wordnet`, `stopwords`), but you can also run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

---

## 🚀 Usage

To run the entire pipeline (Preprocessing → Modeling → Visualization), simply execute the main script:

```bash
python main.py
```

### Execution Flow:

1. **Preprocessing:** Checks for `processed_data.csv`. If missing, it loads the raw CSVs, cleans the text, applies Lemmatization, and saves the processed data.
2. **Modeling:** Trains a Logistic Regression model and a Deep Learning model, then prints a comparison table of their accuracy and F1-scores.
3. **Visualization:** Generates and saves the following images:
   - `sentiment_distribution.png`
   - `wordcloud_positive.png`
   - `wordcloud_negative.png`
   - `sentiment_trend.png`

---

## 📊 Results & Performance

The pipeline outputs a comparison table in the console, highlighting the "Winner" between the models based on accuracy.

Example Output:

```text
=================================================================
Metric               | Logistic Regression  | Deep Learning
-----------------------------------------------------------------
Accuracy             | 0.8540               | 0.8710
Precision            | 0.8420               | 0.8650
Recall               | 0.8540               | 0.8710
F1-Score             | 0.8480               | 0.8680
=================================================================
🏆 Deep Learning performed better by 1.70%
```

---

## 📝 Analysis Report

For a detailed breakdown of the findings, methodology, and strategic recommendations, refer to `report_structure.md`. This document outlines the executive summary, data acquisition process, and business insights derived from the analysis.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to verify the issue or pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
