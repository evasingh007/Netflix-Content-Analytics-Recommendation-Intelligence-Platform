import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Netflix Analytics & Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_netflix_titles.csv")
    with open("sim_matrix.pkl", "rb") as f:
        sim = pickle.load(f)
    return df, sim

df, sim_matrix = load_data()
indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

# Header Section
st.title("🎬 Netflix Content Analytics & Intelligence Platform")
st.markdown("Automated EDA, Predictive Categorization, and Recommendation Engine")

# KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Catalog Titles", f"{len(df):,}")
kpi2.metric("Movies", f"{len(df[df['type']=='Movie']):,}")
kpi3.metric("TV Shows", f"{len(df[df['type']=='TV Show']):,}")
kpi4.metric("Avg Release Year", int(df["release_year"].mean()))

st.divider()

# Main Tabs
tab1, tab2 = st.tabs(["📊 Catalog Distribution Analytics", "🔍 AI Recommendation System"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        type_counts = df["type"].value_counts().reset_index()
        fig_type = px.pie(type_counts, values="count", names="type", title="Movies vs TV Shows Ratio", hole=0.4)
        st.plotly_chart(fig_type, use_container_width=True)
    with col2:
        top_countries = df[df["country"] != "Unknown"]["country"].value_counts().head(10).reset_index()
        fig_country = px.bar(top_countries, x="country", y="count", title="Top 10 Content Producing Countries", color="count")
        st.plotly_chart(fig_country, use_container_width=True)

with tab2:
    st.subheader("Semantic Recommendation Engine")
    selected_title = st.selectbox("Select or type a title:", df["title"].values)
    
    if st.button("Generate Recommendations", type="primary"):
        key = selected_title.strip().lower()
        if key in indices:
            idx = indices[key]
            sim_scores = list(enumerate(sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            
            movie_indices = [i[0] for i in sim_scores]
            scores = [round(i[1] * 100, 2) for i in sim_scores]
            
            recs = df.iloc[movie_indices][["title", "type", "rating", "listed_in", "description"]].copy()
            recs["similarity_score"] = scores
            
            for _, row in recs.iterrows():
                with st.expander(f"⭐ {row['title']} — Match: {row['similarity_score']}%"):
                    st.write(f"**Type:** {row['type']} | **Rating:** {row['rating']} | **Genres:** {row['listed_in']}")
                    st.write(row['description'])
        else:
            st.warning("Title not found.")