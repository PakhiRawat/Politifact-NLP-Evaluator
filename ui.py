# ====================================================
# 💜 UI.PY — FINAL FRONTEND (Compact Scraper + Centered Analysis)
# ====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import backend logic from main.py
from main import (
    scrape_politifact_date_range,
    run_models_and_benchmark,
    google_fact_check_single,
    run_cross_validation,
    API_KEY
)


# ====================================================
# 🧠 SECTION 1: SIDEBAR (Fact Checker + Cross-Validation)
# ====================================================
def render_sidebar(api_key):
    """Render manual Google Fact Checker and cross-validation controls."""
    st.sidebar.header("Google Fact Checker & Validation")

    # --- Manual Fact Checker ---
    st.sidebar.subheader("Manual Fact Checker")
    manual_stmt = st.sidebar.text_area("Enter a statement to verify:", height=100)
    if st.sidebar.button("Check Fact", key="fact_check"):
        if not manual_stmt.strip():
            st.sidebar.warning("Please enter a statement first.")
        else:
            with st.spinner("Checking with Google Fact Checker API..."):
                resp = google_fact_check_single(manual_stmt, api_key)
                if resp is None or "error" in resp:
                    st.sidebar.error("Error fetching results.")
                else:
                    claims = resp.get("claims", [])
                    results = []
                    for claim in claims:
                        for rev in claim.get("claimReview", []):
                            pub = rev.get("publisher", {}).get("name", "")
                            if pub.lower() == "politifact":
                                continue
                            rating = rev.get("textualRating", "")
                            url = rev.get("url", "")
                            results.append((pub, rating, url))
                    if not results:
                        st.sidebar.info("No external (non-PolitiFact) results found.")
                    else:
                        st.sidebar.success("External fact-checks found:")
                        for pub, rating, url in results:
                            st.sidebar.markdown(f"**{pub}** — {rating}")
                            if url:
                                st.sidebar.markdown(f"[Read more]({url})")

    # --- Cross Validation ---
    st.sidebar.subheader("Cross-Validation")
    if "data" not in st.session_state or st.session_state.data is None:
        st.sidebar.info("Scrape data first to enable cross-validation.")
        return None

    total_data = len(st.session_state.data)
    st.sidebar.caption(f"Total scraped statements: {total_data}")

    sample_size_input = st.sidebar.text_input(
        "Enter number of statements to validate (leave blank for all):", value=""
    )

    if st.sidebar.button("Run Cross-Validation", key="cross_val"):
        with st.spinner("Running cross-validation..."):
            if sample_size_input.strip() == "":
                sample_size = "All"
            else:
                try:
                    sample_size = int(sample_size_input)
                except ValueError:
                    st.sidebar.error("Please enter a valid number or leave blank for all.")
                    return

            progress = st.sidebar.progress(0)
            time.sleep(0.3)
            accuracy, err = run_cross_validation(st.session_state.data, sample_size, api_key)
            if err:
                st.sidebar.error(err)
            else:
                st.sidebar.success(f"Agreement Accuracy: **{accuracy}%**")
            progress.progress(100)
            time.sleep(0.5)
            progress.empty()

    return None


# ====================================================
# 📊 SECTION 2: MAIN PANEL (Compact Scraper + NLP Benchmark)
# ====================================================
def render_main_panel():
    """Main content: compact scraper + centered benchmarking results."""
    st.header("Data Sourcing & NLP Analysis")

    # Layout columns: scraper (small) + analysis (wide)
    col1, col2 = st.columns([1.2, 2.8])

    with col1:
        st.markdown("### Scrape PolitiFact")
        if "data" not in st.session_state or st.session_state.data is None or st.session_state.data.empty:
            today = pd.Timestamp.today().date()
            default_start = today.replace(year=today.year - 1)
            start_date = st.date_input("From", default_start)
            end_date = st.date_input("To", today)
            max_pages = st.slider("Max pages", 1, 50, 5)
            if st.button("Scrape", key="scrape_btn"):
                if start_date > end_date:
                    st.error("Start date must be before end date.")
                else:
                    with st.spinner("Scraping PolitiFact..."):
                        df_scraped = scrape_politifact_date_range(start_date, end_date, max_pages=max_pages)
                    if df_scraped.empty:
                        st.warning("No statements found.")
                    else:
                        st.session_state.data = df_scraped
                        st.success(f"Scraped {len(df_scraped)} statements.")
        else:
            st.success(f"Using {len(st.session_state.data)} scraped statements.")
            st.caption("Here’s a quick preview of the scraped dataset:")
            st.dataframe(st.session_state.data.head(), use_container_width=True)

    with col2:
        if "data" in st.session_state and st.session_state.data is not None:
            data = st.session_state.data

            st.markdown("### Filters")
            if "speaker" in data.columns:
                speakers = sorted(data["speaker"].fillna("").unique().tolist())
                sel_speaker = st.selectbox("Filter by speaker:", options=["All"] + speakers)
                if sel_speaker != "All":
                    data = data[data["speaker"] == sel_speaker]
            st.session_state.data = data

            st.markdown("### NLP Phase Selection")
            nlp_phase = st.selectbox("Choose NLP Phase:", ["Lexical", "Syntactic", "Semantic", "Pragmatic", "Discourse"])
            st.caption(f"Selected phase: {nlp_phase}")

            # Run ML Benchmark
            if st.button("Run Model Benchmarking", key="run_model_btn"):
                if data.empty:
                    st.warning("No data available.")
                else:
                    with st.spinner("Training models and evaluating..."):
                        results_df, err = run_models_and_benchmark(data)
                    if err:
                        st.error(err)
                    else:
                        st.success("Model benchmarking complete!")
                        st.session_state.results_df = results_df
                        st.dataframe(results_df, use_container_width=True)

                        # Chart Visualization (purple only)
                        st.subheader("Model Performance Comparison")
                        metric = st.selectbox("Compare Models on:", ["Accuracy (%)", "F1-Score (%)", "Time (s)"])
                        fig, ax = plt.subplots()
                        ax.bar(results_df["Model"], results_df[metric], color="#7209b7")
                        ax.set_xlabel("Models")
                        ax.set_ylabel(metric)
                        ax.set_title(f"Model Comparison: {metric}")
                        st.pyplot(fig)


# ====================================================
# 📝 SECTION 3: FOOTER
# ====================================================
def render_footer():
    st.markdown("---")
    st.caption(
        "💡 *Powered by PolitiFact data & Google Fact Checker API.*  \n"
        "All rights reserved © 2025 | NLP Phase Evaluator"
    )


# ====================================================
# 🚀 SECTION 4: MAIN APP FLOW
# ====================================================
def main():
    st.title("Politifact NLP Phase Evaluator")
    render_sidebar(API_KEY)
    render_main_panel()
    render_footer()


if __name__ == "__main__":
    main()
