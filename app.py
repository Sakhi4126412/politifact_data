# ============================================
# 📌 Streamlit NLP Phase-wise with All Models + SMOTE + Safe Splitting
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate All Models (with SMOTE & Safe Split)
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    # ✅ Safe train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

    # ✅ Apply SMOTE only on training data
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except Exception as e:
        st.warning(f"⚠️ SMOTE failed: {str(e)}")

    # Train each model
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error"

    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase Analysis", layout="wide")
st.title("📊 Rumor Buster")

st.markdown("### 📁 Upload Your CSV")
uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### ⚙️ Configuration")
    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)
    phase = st.selectbox("Select NLP Phase:", [
        "Lexical & Morphological",
        "Syntactic", 
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])
    run_analysis = st.button("Run Analysis", type="primary")

    if run_analysis:
        st.write(f"### 🔍 Analysis: {phase}")
        X = df[text_col].astype(str)
        y = df[target_col]

        # Extract features
        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                      columns=["polarity", "subjectivity"])

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Pragmatic":
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                      columns=pragmatic_words)

        # Evaluate models
        results = evaluate_models(X_features, y)

        # Convert results to DataFrame and handle errors
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df["Accuracy_numeric"] = pd.to_numeric(results_df["Accuracy"], errors="coerce")
        results_df["Accuracy_numeric"] = results_df["Accuracy_numeric"].fillna(0)
        results_df = results_df.sort_values(by="Accuracy_numeric", ascending=False).reset_index(drop=True)
        best_idx = results_df["Accuracy_numeric"].idxmax()

        # Charts
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # Bar chart
        bars = ax1.bar(results_df["Model"], results_df["Accuracy_numeric"], color=colors, alpha=0.9, edgecolor='darkgray', linewidth=1.5)
        bars[best_idx].set_color('#FFD93D')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        for i, val in enumerate(results_df["Accuracy_numeric"]):
            ax1.text(i, val + 1, f"{val:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Performance')
        ax1.set_ylim(0, min(100, max(results_df["Accuracy_numeric"])+15))
        ax1.grid(axis='y', alpha=0.3)

        # Donut chart
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy_numeric"], 
            labels=results_df["Model"], 
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '', 
            startangle=90, 
            colors=colors,
            explode=[0.1 if i==best_idx else 0 for i in range(len(results_df))]
        )
        centre_circle = plt.Circle((0,0),0.7,fc='white')
        ax2.add_artist(centre_circle)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('black')
        ax2.set_title('Performance Distribution')
        plt.tight_layout()
        st.pyplot(fig)

        # Metrics display
        st.write("### 🏆 Operational Benchmarks")
        cols = st.columns(len(results_df))
        for idx, row in results_df.iterrows():
            with cols[idx]:
                if row["Accuracy"] == "Error":
                    st.metric(label=row["Model"], value="Error")
                elif idx == best_idx:
                    st.metric(label=f"🥇 {row['Model']}", value=f"{row['Accuracy_numeric']:.1f}%", delta="Best")
                else:
                    delta = -round(results_df.loc[best_idx,"Accuracy_numeric"] - row["Accuracy_numeric"],1)
                    st.metric(label=row["Model"], value=f"{row['Accuracy_numeric']:.1f}%", delta=f"{delta}%")

        # Results table
        st.write("### 📋 Detailed Results")
        display_df = results_df.copy()
        display_df["Accuracy_display"] = display_df["Accuracy_numeric"].apply(lambda x: f"{x:.1f}%" if x !=0 else "Error")
        display_df["Rank"] = range(1, len(display_df)+1)
        st.dataframe(display_df[["Rank","Model","Accuracy_display"]], use_container_width=True)

else:
    st.info("👆 Upload a CSV file to start analysis.")

# ============================
# Styling
# ============================
st.markdown("""
<style>
.stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
div[data-testid="metric-container"] { background-color:#f0f2f6; padding:10px; border-radius:10px; border-left:4px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)
