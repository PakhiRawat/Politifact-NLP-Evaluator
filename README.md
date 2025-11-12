# NLP Phase Evaluator – Fact Check and Analysis App

## Overview
This project is a **Streamlit-based NLP analysis and fact-checking tool** designed to:

1. Scrape political statements from [PolitiFact](https://www.politifact.com/).
2. Run NLP model benchmarking across multiple classifiers.
3. Validate the accuracy of scraped statements using the **Google Fact Check API**.
4. Provide both automated and manual fact-checking features.

It helps users analyze NLP phase performance, compare model metrics, and evaluate the truthfulness of statements using verified external sources.

---

## Features

### 1. Data Scraping
- Automatically scrapes recent statements from PolitiFact.
- Configurable by date range and number of pages (up to 50 pages).
- Cleans, structures, and stores the dataset in memory for analysis.

### 2. NLP Model Benchmarking
- Compares multiple machine learning models:
  - Naive Bayes  
  - Decision Tree  
  - Logistic Regression  
  - SVM  
  - KNN
- Displays model accuracy, F1-score, and time performance.
- Visualizes results through comparative bar charts.

### 3. Google Fact Checker Integration
- Uses the **Google Fact Check Tools API** to verify statements.
- Includes:
  - **Manual fact checking** (enter any custom statement).
  - **Cross-validation** (randomly sample or validate all scraped statements).
- Calculates overall agreement accuracy between PolitiFact and other fact-checking publishers.

### 4. Compact and Clean UI
- Built entirely with **Streamlit**.
- Minimal scrolling layout:
  - Compact scraper section.
  - Center-aligned benchmarking and analysis panels.
- Sidebar includes manual and automated fact-checking tools.

---

## Project Structure
```plaintext
project/
│
├── main.py          # Backend logic: scraping, ML models, Google API, validation
├── ui.py            # Frontend Streamlit app: layout, inputs, and visuals
├── requirements.txt # Dependencies for the project
└── README.md        # Documentation
