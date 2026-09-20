# 🎬 Netflix Content Analytics & Recommendation Intelligence Platform
🔗 **Live Interactive Application:** [https://netflix-content-analytics-recommendation-intelligence-platform.streamlit.app/]

An end-to-end data analytics, machine learning, and business intelligence platform built on 8,800+ Netflix records. This project features an automated Python ETL pipeline[cite: 1], SQL Server analytical warehousing[cite: 1], natural language processing (TF-IDF & Cosine Similarity)[cite: 1], supervised machine learning, and interactive dashboards (Streamlit & Microsoft Power BI)[cite: 1].

---

## 📌 Executive Summary & Key Metrics
- **Catalog Scale Analyzed:** 8,797 titles (6,131 Movies | 2,666 TV Shows).
- **Core Library Split:** 69.7% Movies vs. 30.3% TV Shows[cite: 1].
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
  - **Microsoft Power BI:** Dark-themed executive business intelligence suite tracking catalog velocity, volume distribution, and regional share.

---

## 📊 SQL Analytics Overview

All analytical database scripts are stored in [`netflix_sql_analytics.sql`](./netflix_sql_analytics.sql)[cite: 1]. The warehouse queries cover:
- **Year-over-Year (YoY) Library Velocity:** Calculating annual catalog growth rates using the `LAG()` window function[cite: 1].
- **Regional Dominance:** Identifying top content categories and ratings by country using partitioned `DENSE_RANK()` window functions[cite: 1].
- **Catalog Distribution:** Aggregate percentage share and volume splits across media types and production regions.

---

## 📈 Executive BI Dashboard (Power BI Desktop)

A dark-themed business intelligence dashboard designed to complement the analytical pipeline[cite: 1, 2]. Built directly against the local SQL Server warehouse table (`cleaned_netflix_titles`)[cite: 2] and saved as [`Netflix_Dashboard.pbix`](./Netflix_Dashboard.pbix)[cite: 10].

![Netflix Executive Dashboard]
<img width="1316" height="742" alt="image" src="https://github.com/user-attachments/assets/8afe8704-d7d0-4455-8a9c-25eb4d286ac4" />


### Key DAX Measures
* **Total Titles (8,797)**[cite: 1, 3]:  
  ```dax
  Total Titles = COUNTROWS('cleaned_netflix_titles')
  ```[cite: 13]
  ├── .venv/                         # Virtual environment (ignored in git)
├── assets/
│   └── powerbi_dashboard.png      # Power BI dashboard preview image
├── cleaned_netflix_titles.csv     # Standardized & cleaned dataset
├── data_pipeline.py               # Automated ETL & data transformation script
├── train_models.py                # Recommender matrix & ML training pipeline
├── app.py                         # Streamlit interactive application
├── netflix_sql_analytics.sql      # Production SQL queries & window functions
├── requirements.txt               # Locked environment dependencies
├── Netflix_Dashboard.pbix         # Power BI executive report file
└── README.md                      # Project documentation
* **Total Movies (6,131)**[cite: 1, 3]:  
  ```dax
  Total Movies = CALCULATE(COUNTROWS('cleaned_netflix_titles'), 'cleaned_netflix_titles'[type] = "Movie")
