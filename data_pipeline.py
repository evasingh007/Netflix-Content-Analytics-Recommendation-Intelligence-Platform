import pandas as pd
import numpy as np

def clean_netflix_data(input_path: str = 'netflix_titles.csv', output_path: str = 'cleaned_netflix_titles.csv'):
    print(f"Loading raw dataset from '{input_path}'...")
    df = pd.read_csv(input_path)
    print(f"Raw records loaded: {len(df)}")
    
    # 1. Handle missing values in categorical fields
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['rating'] = df['rating'].fillna('Not Rated')
    
    # 2. Drop rows missing critical dates (preserves ~8,800 records)
    df = df.dropna(subset=['date_added']).copy()
    
    # 3. Date parsing & feature extraction
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='mixed')
    df['year_added'] = df['date_added'].dt.year.astype(int)
    df['month_added'] = df['date_added'].dt.month_name()
    
    # 4. Text cleaning for NLP features
    df['clean_title'] = df['title'].astype(str).str.strip()
    df['clean_description'] = df['description'].fillna('').astype(str).str.strip()
    df['listed_in'] = df['listed_in'].fillna('').astype(str).str.strip()
    
    # Combine metadata into a unified feature string for TF-IDF / NLP
    df['nlp_metadata'] = (
        df['clean_description'] + " " +
        df['listed_in'] + " " +
        df['director'] + " " +
        df['rating']
    ).str.lower()
    
    # 5. Export cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Data pipeline complete. {len(df)} records validated and saved to '{output_path}'.")
    return df

if __name__ == '__main__':
    clean_netflix_data('netflix_titles.csv', 'cleaned_netflix_titles.csv')