# 🎬 Netflix Content Analytics & Recommendation Intelligence Platform
[https://netflix-content-analytics-recommendation-intelligence-platform.streamlit.app/]

An end-to-end data analytics, machine learning, and business intelligence platform built on 8,800+ Netflix records[cite: 1]. This project features an automated Python ETL pipeline[cite: 1], SQL Server analytical warehousing[cite: 1], natural language processing (TF-IDF & Cosine Similarity)[cite: 1], supervised machine learning, and interactive dashboards (Streamlit & Microsoft Power BI)[cite: 1].

---

## 📌 Executive Summary & Key Metrics
- **Catalog Scale Analyzed:** 8,797 titles (6,131 Movies | 2,666 TV Shows).
- **Core Library Split:** 69.7% Movies vs. 30.3% TV Shows.
- **Top Content Producers:** United States (3,642 titles), India (972 titles), United Kingdom (418 titles).
- **Predictive Modeling:** Random Forest classifier trained on catalog metadata and plot synopses to evaluate production origins and catalog distributions.
- **Semantic Recommender:** NLP-driven cosine similarity vector matching delivering sub-second recommendations across multi-genre titles[cite: 1].

---

## 🛠️ Technical Stack & Architecture
- **Data Engineering & ETL:** Python (Pandas, NumPy, RegEx) — automated missing-value handling, type normalization, and metadata engineering.
- **Data Warehousing & Querying:** Microsoft SQL Server (SSMS), T-SQL, Common Table Expressions (CTEs), and Window Functions (`LAG`, `DENSE_RANK`)[cite: 1].
- **Machine Learning & NLP:** Scikit-Learn — `TfidfVectorizer`, Cosine Similarity matrix serialization, Random Forest Classification[cite: 1].
- **Dashboards & Visuals:**
  - **Streamlit + Plotly:** Interactive web application for catalog exploration and real-time semantic recommendations[cite: 1].
  - **Microsoft Power BI:** Analytical dashboard tracking library growth, content share, and regional distributions[cite: 1].

---

## 📊 SQL Analytics Overview

All analytical database scripts are stored in [`netflix_sql_analytics.sql`](./netflix_sql_analytics.sql)[cite: 1]. The warehouse queries cover:
- **Year-over-Year (YoY) Library Velocity:** Calculating annual catalog growth rates using the `LAG()` window function[cite: 1].
- **Regional Dominance:** Identifying top content categories and ratings by country using partitioned `DENSE_RANK()` window functions[cite: 1].
- **Catalog Distribution:** Aggregate percentage share and volume splits across media types and production regions.

---

## 🤖 Machine Learning & Recommendation Engine

1. **Semantic Content Recommender (`sim_matrix.pkl`):**
   - Combines clean synopses, genre tags, and cast metadata using `TfidfVectorizer` (5,000 features, English stop words).
   - Computes pairwise cosine similarity across all 8,797 titles, generating instant similarity-ranked recommendations via Streamlit.

2. **Catalog Classification (`rf_model.pkl`):**
   - Random Forest model trained on multi-feature text embeddings and catalog attributes.
   - Evaluated using standard classification metrics (precision, recall, and F1-score) to predict catalog patterns.

---

## 📁 Repository Structure

```text
├── .venv/                         # Virtual environment (ignored in git)
├── cleaned_netflix_titles.csv     # Standardized & cleaned dataset
├── data_pipeline.py               # Automated ETL & data transformation script
├── train_models.py                # Recommender matrix & ML training pipeline
├── app.py                         # Streamlit interactive application
├── netflix_sql_analytics.sql      # Production SQL queries & window functions
├── requirements.txt               # Locked environment dependencies
├── Netflix_Dashboard.pbix         # Power BI report file
└── README.md                      # Project documentation
