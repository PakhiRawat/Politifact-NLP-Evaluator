# ====================================================
# 🧠 MAIN.PY — BACKEND LOGIC ONLY (Scraping, ML, API, Cross-Validation)
# ====================================================

# ------------------------------
# 🎯 SECTION 1: IMPORTS & CONFIG
# ------------------------------
import pandas as pd
import numpy as np
import time
import re
import requests
from bs4 import BeautifulSoup
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import random

# ------------------------------
# 🔑 SECTION 2: CONFIG VARIABLES
# ------------------------------
API_KEY = "AIzaSyD5iO1Lcpr4C0f06iDFOkMVUVSLaVnaAzA"       # Replace with your actual key
SCRAPE_SLEEP = 0.6                  # seconds between scraping requests
GOOGLE_SLEEP = 0.15                 # seconds between API requests


# ====================================================
# ⚙️ SECTION 3: UTILITY FUNCTIONS
# ====================================================
def extract_date_from_text(text):
    """Extract date from messy PolitiFact HTML text."""
    if not text or not isinstance(text, str):
        return pd.NaT
    m = re.search(r'([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})', text)
    if not m:
        m = re.search(r'([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})', text)
    if m:
        try:
            return pd.to_datetime(m.group(1), errors='coerce')
        except Exception:
            pass
    m2 = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m2:
        return pd.to_datetime(m2.group(1), errors='coerce')
    return pd.to_datetime(text, errors='coerce')


def normalize_rating_textual(rating_text):
    """Normalize textual rating to 'true', 'false', 'mixed', 'other'."""
    if not isinstance(rating_text, str):
        return "other"
    r = rating_text.strip().lower()
    true_terms = ["true", "mostly true", "mostly-true"]
    false_terms = ["false", "mostly false", "mostly-false", "pants on fire"]
    mixed_terms = ["half true", "half-true", "mixture", "partly true"]

    for t in true_terms:
        if t in r:
            return "true"
    for f in false_terms:
        if f in r:
            return "false"
    for m in mixed_terms:
        if m in r:
            return "mixed"
    if "true" in r:
        return "true"
    if "false" in r:
        return "false"
    return "other"


# ====================================================
# 📰 SECTION 4: POLITIFACT SCRAPER (cached externally)
# ====================================================
def scrape_politifact_date_range(start_dt: date, end_dt: date, max_pages: int = 10, sleep_sec: float = SCRAPE_SLEEP):
    """Scrape PolitiFact fact checks within given date range."""
    base_url = "https://www.politifact.com/factchecks/list/?page="
    headers = {"User-Agent": "Mozilla/5.0"}
    all_rows = []
    start_ts = pd.to_datetime(start_dt).normalize()
    end_ts = pd.to_datetime(end_dt).normalize()

    for page in range(1, max_pages + 1):
        url = base_url + str(page)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            break

        cards = soup.select("div.m-statement") or soup.select("li.o-listicle__item") or soup.select("article")
        if not cards:
            break

        page_dates = []
        for card in cards:
            statement = None
            quote = card.select_one(".m-statement__quote") or card.select_one("p") or card.select_one("div.quote")
            if quote:
                statement = quote.get_text(separator=" ", strip=True)
            speaker = None
            sp = card.select_one(".m-statement__name") or card.select_one("a")
            if sp:
                speaker = sp.get_text(strip=True)
            rating = None
            img = card.select_one(".m-statement__meter img")
            if img and img.has_attr("alt"):
                rating = img["alt"].strip()
            else:
                meter = card.select_one(".m-statement__meter") or card.select_one(".meter")
                if meter:
                    rating = meter.get_text(strip=True)
            date_text = None
            footer = card.select_one(".m-statement__footer") or card.select_one("footer")
            if footer:
                date_text = footer.get_text(separator=" ", strip=True)
            if not date_text:
                t = card.find("time")
                if t:
                    date_text = t.get_text(strip=True)
            parsed_date = extract_date_from_text(date_text)
            if parsed_date is pd.NaT or pd.isna(parsed_date):
                continue
            parsed_date = pd.to_datetime(parsed_date).normalize()
            page_dates.append(parsed_date)
            if start_ts <= parsed_date <= end_ts:
                all_rows.append({
                    "statement": statement or "",
                    "speaker": speaker or "",
                    "rating": rating or "",
                    "date": parsed_date.date().isoformat()
                })
        if page_dates and max(page_dates) < start_ts:
            break
        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ====================================================
# 🤖 SECTION 5: MODEL TRAINING & BENCHMARKING
# ====================================================
def run_models_and_benchmark(data):
    """Train and evaluate multiple ML models on the dataset."""
    data = data.copy()
    data.columns = data.columns.str.strip().str.lower()
    label_candidates = ["label", "rating", "target", "class"]
    y_col = next((c for c in label_candidates if c in data.columns), None)
    if "statement" not in data.columns or y_col is None:
        return None, "Data must have 'statement' and a label column."

    X = data["statement"].astype(str)
    y = data[y_col].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y_encoded, test_size=0.2, random_state=42,
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear"),
        "KNN": KNeighborsClassifier()
    }

    results = []
    for name, model in models.items():
        start_t = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_t = time.time()

        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average="weighted") * 100
        duration = end_t - start_t

        results.append({
            "Model": name,
            "Accuracy (%)": round(acc, 2),
            "F1-Score (%)": round(f1, 2),
            "Time (s)": round(duration, 3)
        })
    return pd.DataFrame(results), None


# ====================================================
# 🌐 SECTION 6: GOOGLE FACT CHECK API BACKEND
# ====================================================
def google_fact_check_single(statement, api_key=API_KEY):
    """Query Google Fact Check API for a single statement."""
    if not statement or not api_key:
        return None
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": statement, "key": api_key, "languageCode": "en"}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ====================================================
# 🔍 SECTION 7: CROSS-VALIDATION BACKEND
# ====================================================
def run_cross_validation(data, sample_size, api_key=API_KEY):
    """
    Randomly sample 'sample_size' statements from data and
    compute agreement accuracy with Google Fact Checker API.
    """
    if data is None or data.empty:
        return None, "No data available for cross-validation."

    statements = data["statement"].astype(str).tolist()
    total_available = len(statements)

    # if 'All' selected, validate all scraped data
    if sample_size == "All":
        sample_statements = statements
    else:
        sample_size = min(int(sample_size), total_available)
        sample_statements = random.sample(statements, sample_size)

    records = []
    for stmt in sample_statements:
        resp = google_fact_check_single(stmt, api_key)
        time.sleep(GOOGLE_SLEEP)
        if resp is None or "error" in resp:
            continue

        claims = resp.get("claims", [])
        ext_ratings = []
        for claim in claims:
            for rev in claim.get("claimReview", []):
                pub = rev.get("publisher", {}).get("name", "").lower().strip()
                if pub == "politifact":  # skip same source
                    continue
                norm = normalize_rating_textual(rev.get("textualRating", ""))
                ext_ratings.append(norm)
        if not ext_ratings:
            continue

        ext_mode = Counter(ext_ratings).most_common(1)[0][0]
        politifact_raw = data.loc[data["statement"] == stmt, "rating"].iloc[0] if "rating" in data.columns else ""
        politifact_norm = normalize_rating_textual(str(politifact_raw))
        agreement_flag = (ext_mode == politifact_norm)
        records.append({"statement": stmt, "agreement": agreement_flag})

    if not records:
        return None, "No external fact-checks found for selected statements."

    df_rec = pd.DataFrame(records)
    accuracy = (df_rec["agreement"].sum() / len(df_rec)) * 100
    return round(accuracy, 2), None


# ====================================================
# 🚀 SECTION 8: MAIN ENTRYPOINT (TO CONNECT TO UI)
# ====================================================
def main():
    """
    Placeholder main function — UI elements will be built in ui.py
    and call these backend functions.
    """
    print("Backend ready. Connect via ui.py for full Streamlit app.")


if __name__ == "__main__":
    main()
