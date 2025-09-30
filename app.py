# ============================================
# 📌 Streamlit NLP Phase-wise with All Models + SMOTE
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

from imblearn.over_sampling import SMOTE   # ✅ NEW

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
    """Tokenization + Stopwords removal + Lemmatization"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """Part-of-Speech tags"""
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity & subjectivity"""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count + first word of each sentence"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    """Counts of modality & special words"""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate All Models (with SMOTE)
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Apply SMOTE only on training data
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except Exception as e:
        st.warning(f"⚠️ SMOTE failed: {str(e)}")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase Analysis", layout="wide")

st.title("📊 Rumor Buster")

# File upload in the center (not sidebar)
st.markdown("### 📁 Data, Assemble!")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("Awesome!!File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

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

    run_analysis = st.button("Click here for Analysis", type="primary")

# Main content area
if uploaded_file:
    st.write("### 📋 Data Display")
    st.dataframe(df.head(), use_container_width=True)
    
    if run_analysis:
        st.write("---")
        st.write(f"### 🔍 Investigation: {phase}")
        
        with st.spinner("Wait a sec… the universe is rearranging itself for you."):
            X = df[text_col].astype(str)
            y = df[target_col]

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

            # Run all models (with SMOTE balancing)
            results = evaluate_models(X_features, y)

        # Convert results to DataFrame
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        # Display results
        st.write("---")
        st.subheader("📊 Brainpower Breakdown")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Enhanced Bar Chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(results_df["Model"], results_df["Accuracy"], 
                      color=colors, alpha=0.9, edgecolor='darkgray', linewidth=1.5)
        
        # Highlight the best model
        best_idx = results_df["Accuracy"].idxmax()
        bars[best_idx].set_color('#FFD93D')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)

        # Add value labels on bars
        for i, (model, acc) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Model Performance - {phase}\n', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, min(100, max(results_df["Accuracy"]) + 15))
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=15)
        
        # Donut Chart for performance distribution
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"], 
            labels=results_df["Model"], 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0.1 if i == best_idx else 0 for i in range(len(results_df))]
        )
        # Add white circle for donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax2.add_artist(centre_circle)

        # Enhance text
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax2.set_title('Performance Distribution (Donut Chart)\n', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display metrics in a row
        st.write("### 🏆Operational Benchmarks")
        cols = st.columns(4)
        for idx, (model, accuracy) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            with cols[idx]:
                if idx == best_idx:
                    st.metric(
                        label=f"🥇 {model}",
                        value=f"{accuracy:.1f}%",
                        delta="Best Performance"
                    )
                else:
                    st.metric(
                        label=model,
                        value=f"{accuracy:.1f}%",
                        delta=f"{-round(accuracy - results_df.loc[best_idx, 'Accuracy'], 1)}%"
                    )
        
        # Detailed results table
        st.write("### 📋 The Inside Scoop")
        results_display = results_df.copy()
        results_display["Accuracy"] = results_display["Accuracy"].apply(lambda x: f"{x:.1f}%")
        results_display["Rank"] = range(1, len(results_display) + 1)
        results_display = results_display[["Rank", "Model", "Accuracy"]]
        st.dataframe(results_display, use_container_width=True)

else:
    st.info("👆 Bring your CSV, we’ll do the heavy lifting.")

# ============================
# Styling
# ============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)
