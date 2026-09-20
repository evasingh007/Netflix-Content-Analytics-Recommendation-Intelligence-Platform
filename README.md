# 🎬 Netflix Content Analytics & Recommendation Intelligence Platform
🔗 **Live Interactive Application:** [https://netflix-content-analytics-recommendation-intelligence-platform.streamlit.app/]

An end-to-end data analytics, machine learning, and business intelligence platform built on 8,800+ Netflix records. This project features an automated Python ETL pipeline, SQL Server analytical warehousing, natural language processing (TF-IDF & Cosine Similarity), supervised machine learning, and interactive dashboards (Streamlit & Microsoft Power BI).

---

## 📌 Executive Summary & Key Metrics
- **Catalog Scale Analyzed:** 8,797 titles (6,131 Movies | 2,666 TV Shows).
- **Core Library Split:** 69.7% Movies vs. 30.3% TV Shows.
- **Top Content Producers:** United States (3,642 titles), India (972 titles), United Kingdom (418 titles).
- **Predictive Modeling:** Random Forest classifier trained on catalog metadata and plot synopses to evaluate production origins and catalog distributions.
- **Semantic Recommender:** NLP-driven cosine similarity vector matching delivering sub-second recommendations across multi-genre titles.

---

## 🛠️ Technical Stack & Architecture
- **Data Engineering & ETL:** Python (Pandas, NumPy, RegEx) — automated missing-value handling, type normalization, and metadata engineering.
- **Data Warehousing & Querying:** Microsoft SQL Server (SSMS), T-SQL, Common Table Expressions (CTEs), and Window Functions (`LAG`, `DENSE_RANK`).
- **Machine Learning & NLP:** Scikit-Learn — `TfidfVectorizer`, Cosine Similarity matrix serialization, Random Forest Classification.
- **Dashboards & Visuals:**
  - **Streamlit + Plotly:** Interactive web application for catalog exploration and real-time semantic recommendations.
  - **Microsoft Power BI:** Dark-themed executive business intelligence suite tracking catalog velocity, volume distribution, and regional share.

---

## 📊 SQL Analytics Overview

All analytical database scripts are stored in [`netflix_sql_analytics.sql`](./netflix_sql_analytics.sql). The warehouse queries cover:
- **Year-over-Year (YoY) Library Velocity:** Calculating annual catalog growth rates using the `LAG()` window function.
- **Regional Dominance:** Identifying top content categories and ratings by country using partitioned `DENSE_RANK()` window functions.
- **Catalog Distribution:** Aggregate percentage share and volume splits across media types and production regions.

---

## 📈 Executive BI Dashboard (Power BI Desktop)

A dark-themed business intelligence dashboard designed to complement the analytical pipeline. Built directly against the local SQL Server warehouse table (`cleaned_netflix_titles`) and saved as [`Netflix_Dashboard.pbix`](./Netflix_Dashboard.pbix).

![Netflix Executive Dashboard](assets/powerbi_dashboard.png)

### Key DAX Measures
* **Total Titles (8,797)**:
  ```dax
  Total Titles = COUNTROWS('cleaned_netflix_titles')
