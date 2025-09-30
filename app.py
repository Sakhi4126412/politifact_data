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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ============================
# Train & Evaluate All Models (with SMOTE + Full Metrics)
# ============================
def evaluate_models(X_features, y):
    results = []
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Apply SMOTE on training data
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
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

            results.append({
                "Model": name,
                "Accuracy": round(acc, 2),
                "Precision": round(prec, 2),
                "Recall": round(rec, 2),
                "F1-Score": round(f1, 2)
            })
        except Exception as e:
            results.append({
                "Model": name,
                "Accuracy": f"Error: {str(e)}",
                "Precision": "-",
                "Recall": "-",
                "F1-Score": "-"
            })

    return pd.DataFrame(results)


# ✅ Rest of your Streamlit UI code remains SAME
