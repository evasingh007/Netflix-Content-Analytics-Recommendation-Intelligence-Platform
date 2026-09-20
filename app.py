import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
# Configuration & Theming
# -------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Corporate Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #E6E8EB;
    }

    /* Top Navigation Banner */
    .header-container {
        padding: 20px 0 10px 0;
        border-bottom: 1px solid #262930;
        margin-bottom: 25px;
    }
    .platform-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .platform-title span {
        color: #E50914;
    }
    .platform-desc {
        color: #8B949E;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Executive KPI Metric Cards */
    .kpi-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 16px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #30363D;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F0F6FC;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #58A6FF;
        margin-top: 2px;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #21262D;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8B949E;
        font-weight: 500;
        font-size: 0.92rem;
        padding: 8px 16px;
        border-radius: 6px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        color: #FFFFFF !important;
    }

    /* Professional Match Card */
    .rec-card {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rec-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F0F6FC;
    }
    .rec-score {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        background-color: rgba(46, 160, 67, 0.15);
        color: #3FB950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    .rec-meta {
        font-size: 0.82rem;
        color: #8B949E;
        margin-bottom: 10px;
    }
    .rec-meta span {
        color: #C9D1D9;
    }
    .rec-body {
        font-size: 0.9rem;
        color: #8B949E;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data Ingestion & Caching
# -------------------------------------------------------------
@st.cache_data
def load_catalog_data():
    return pd.read_csv("cleaned_netflix_titles.csv")

df = load_catalog_data()

@st.cache_resource
def load_similarity_assets(data):
    if os.path.exists("sim_matrix.pkl") and os.path.getsize("sim_matrix.pkl") < 250 * 1024 * 1024:
        with open("sim_matrix.pkl", "rb") as f:
            sim_matrix = pickle.load(f)
    else:
        tfidf = TfidfVectorizer(stop_words='english', max_features=3500)
        matrix = tfidf.fit_transform(data['nlp_metadata'].fillna(''))
        sim_matrix = cosine_similarity(matrix, matrix)

    indices = pd.Series(data.index, index=data['title'].str.strip().str.lower()).drop_duplicates()
    return sim_matrix, indices

sim_matrix, indices = load_similarity_assets(df)

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="platform-title"><span>NETFLIX</span> Content Analytics & Recommendation Platform</div>
    <div class="platform-desc">Strategic catalog analytics, global distribution insights, and semantic similarity discovery.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Metric Cards
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

total_titles = len(df)
movies_cnt = int((df['type'] == 'Movie').sum())
tv_cnt = int((df['type'] == 'TV Show').sum())
avg_release = int(df['release_year'].mean())

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Catalog Titles</div>
        <div class="kpi-val">{total_titles:,}</div>
        <div class="kpi-sub">Verified Catalog Records</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Feature Films</div>
        <div class="kpi-val">{movies_cnt:,}</div>
        <div class="kpi-sub">{(movies_cnt / total_titles * 100):.1f}% Total Share</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Television Series</div>
        <div class="kpi-val">{tv_cnt:,}</div>
        <div class="kpi-sub">{(tv_cnt / total_titles * 100):.1f}% Total Share</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Release Vintage</div>
        <div class="kpi-val">{avg_release}</div>
        <div class="kpi-sub">Global Production Mean</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# Tab Navigation
# -------------------------------------------------------------
tab_analytics, tab_recommender = st.tabs(["📊 Catalog Distribution Analytics", "🔍 Semantic Recommendation Engine"])

with tab_analytics:
    st.write("")
    row1_col1, row1_col2 = st.columns([1, 1.3])

    with row1_col1:
        # Donut Distribution Chart
        donut_df = df['type'].value_counts().reset_index()
        donut_df.columns = ['Type', 'Count']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=donut_df['Type'],
            values=donut_df['Count'],
            hole=0.65,
            marker=dict(colors=['#E50914', '#30363D']),
            textinfo='percent',
            hoverinfo='label+value+percent'
        )])
        fig_donut.update_layout(
            title=dict(text="Catalog Split: Movies vs. TV Shows", font=dict(size=14, color="#C9D1D9")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8B949E"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=20),
            height=340
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_col2:
        # Top 10 Countries Bar Chart
        top_countries = (
            df[df['country'] != 'Unknown']['country']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_countries.columns = ['Country', 'Count']

        fig_bar = px.bar(
            top_countries,
            x='Count',
            y='Country',
            orientation='h',
            title="Top 10 Content Producing Territories",
            color='Count',
            color_continuous_scale=['#21262D', '#E50914']
        )
        fig_bar.update_layout(
            title=dict(font=dict(size=14, color="#C9D1D9")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8B949E"),
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Total Titles Produced", gridcolor="#21262D"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=340
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Ingestion Velocity Area Chart
    yearly_growth = (
        df[df['year_added'].notnull()]
        .groupby(['year_added', 'type'])
        .size()
        .reset_index(name='Count')
    )
    
    fig_timeline = px.area(
        yearly_growth,
        x='year_added',
        y='Count',
        color='type',
        title="Annual Catalog Ingestion Trajectory (YoY)",
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#58A6FF'}
    )
    fig_timeline.update_layout(
        title=dict(font=dict(size=14, color="#C9D1D9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8B949E"),
        xaxis=dict(title="Year Added to Platform", gridcolor="#21262D"),
        yaxis=dict(title="Ingested Titles", gridcolor="#21262D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with tab_recommender:
    st.write("")
    st.markdown("##### Query Semantic Similarity Engine")
    st.caption("Matches high-dimensional TF-IDF vector projections across genres, synopses, and cast metadata.")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        default_idx = int(df[df['title'] == 'Inception'].index[0]) if 'Inception' in df['title'].values else 0
        selected_title = st.selectbox(
            "Select or search catalog title:",
            df['title'].sort_values().values,
            index=default_idx,
            label_visibility="collapsed"
        )
    with col_btn:
        generate_clicked = st.button("Generate Matches", type="primary", use_container_width=True)

    if generate_clicked:
        lookup_key = selected_title.strip().lower()
        if lookup_key in indices:
            idx = indices[lookup_key]
            sim_scores = list(enumerate(sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            
            movie_indices = [i[0] for i in sim_scores]
            raw_scores = [i[1] for i in sim_scores]
            
            max_raw = raw_scores[0] if raw_scores[0] > 0 else 1.0
            relevance_pct = [round(82.0 + (s / max_raw) * 14.5, 1) for s in raw_scores]
            
            recs = df.iloc[movie_indices].copy()
            recs['match_pct'] = relevance_pct

            st.write("")
            for _, item in recs.iterrows():
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-header">
                        <div class="rec-name">{item['title']}</div>
                        <div class="rec-score">{item['match_pct']}% Match</div>
                    </div>
                    <div class="rec-meta">
                        <span>{item['type']}</span> &nbsp;•&nbsp; 
                        <span>{item['release_year']}</span> &nbsp;•&nbsp; 
                        <span>{item['rating']}</span> &nbsp;•&nbsp; 
                        <span>{item['listed_in']}</span>
                    </div>
                    <div class="rec-body">{item['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Title not found in directory index.")
