import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Loading cleaned dataset...")
df = pd.read_csv("cleaned_netflix_titles.csv")

# Filter out rows where country is Unknown
df_model = df[df['country'] != 'Unknown'].copy()

# -------------------------------------------------------------
# 1. NLP: TF-IDF Vectorizer & Cosine Similarity Matrix
# -------------------------------------------------------------
print("Building TF-IDF Vectorizer & Cosine Similarity Matrix...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['nlp_metadata'])

sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

with open("sim_matrix.pkl", "wb") as f:
    pickle.dump(sim_matrix, f)

print(f"Similarity matrix generated and saved: {sim_matrix.shape}")

# -------------------------------------------------------------
# 2. ML: Content Origin Classification (US Domestic vs International)
# -------------------------------------------------------------
print("\nTraining Random Forest Classifier on Content Origin...")

# Target: 1 for US production, 0 for International production
y = df_model['country'].apply(lambda x: 1 if 'United States' in str(x) else 0)

# Feature text: plot description + genre tags + rating context (strictly omitting country names)
feature_text = (
    df_model['clean_description'].fillna('') + ' ' +
    df_model['listed_in'].fillna('').str.replace('International Movies', '', regex=False).str.replace('International TV Shows', '', regex=False) + ' ' +
    df_model['rating'].fillna('')
)

clf_tfidf = TfidfVectorizer(stop_words='english', max_features=3000, min_df=2)
X = clf_tfidf.fit_transform(feature_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=22,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nFinal Random Forest Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['International', 'US Domestic']))

with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model successfully saved and serialized.")