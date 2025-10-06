# ============================================
# 📰 Streamlit Fake News NLP Analysis + SMOTE + Model Recommendation (with Logging)
# ============================================

import streamlit as st
import pandas as pd
import spacy
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
from wordcloud import WordCloud

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(
    filename="app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Load SpaCy Model
# -------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error("❌ Failed to load SpaCy model. Please run: `python -m spacy download en_core_web_sm`")
    logging.error(f"SpaCy model loading error: {e}")
    st.stop()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="📰 Fake News NLP Analysis", layout="wide")
st.title("📰 Fake News Detection & NLP Phase Analysis")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=['csv'])
if not uploaded_file:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# -------------------------------
# Load Dataset
# -------------------------------
try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
except Exception as e:
    st.error("❌ Error loading CSV file. Please ensure it’s a valid CSV format.")
    logging.error(f"CSV Loading Error: {e}")
    st.stop()

# -------------------------------
# Feature & Target Selection
# -------------------------------
feature_col = st.selectbox("Select Text Column", df.columns)
target_col = st.selectbox("Select Target Column", df.columns)

# -------------------------------
# NLP Phase-wise Analysis Tabs
# -------------------------------
tabs = st.tabs(["Lexical/Morphological", "Syntactic", "Semantic", "Pragmatic", "Discourse", "Target Distribution"])

# ----- Lexical & Morphological -----
with tabs[0]:
    st.markdown("### 1️⃣ Lexical & Morphological Analysis")
    df['word_count'] = df[feature_col].astype(str).apply(lambda x: len(x.split()))
    df['char_count'] = df[feature_col].astype(str).apply(len)

    col1, col2 = st.columns(2)
    fig1 = px.histogram(df, x='word_count', nbins=20, title="Word Count Distribution")
    fig2 = px.histogram(df, x='char_count', nbins=20, title="Character Count Distribution")
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# ----- Syntactic -----
with tabs[1]:
    st.markdown("### 2️⃣ Syntactic Analysis (POS Tags)")
    try:
        def pos_tags(text):
            doc = nlp(str(text))
            return [token.pos_ for token in doc]
        df['pos_tags'] = df[feature_col].astype(str).apply(pos_tags)
        pos_list = [tag for sublist in df['pos_tags'] for tag in sublist]
        pos_df = pd.Series(pos_list).value_counts().reset_index()
        pos_df.columns = ['POS', 'Count']
        fig_pos = px.bar(pos_df, x='POS', y='Count', title="POS Tag Distribution", color='Count')
        st.plotly_chart(fig_pos, use_container_width=True)
    except Exception as e:
        logging.error(f"Syntactic Analysis Error: {e}")
        st.error("❌ Failed to perform POS tagging. Please verify your text data.")

# ----- Semantic -----
with tabs[2]:
    st.markdown("### 3️⃣ Semantic Analysis (WordCloud)")
    try:
        text = " ".join(df[feature_col].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array())
    except Exception as e:
        logging.error(f"WordCloud Generation Error: {e}")
        st.error("❌ Failed to generate word cloud. Please ensure text column is valid.")

# ----- Pragmatic -----
with tabs[3]:
    st.markdown("### 4️⃣ Pragmatic Analysis (Sentence Length)")
    df['sentence_length'] = df[feature_col].astype(str).apply(lambda x: len(x.split()))
    fig_prag = px.histogram(df, x='sentence_length', nbins=20, title="Sentence Length Distribution")
    st.plotly_chart(fig_prag, use_container_width=True)

# ----- Discourse -----
with tabs[4]:
    st.markdown("### 5️⃣ Discourse Integration (Class-wise WordClouds)")
    try:
        for val in df[target_col].unique():
            st.markdown(f"**Class: {val}**")
            subset_text = " ".join(df[df[target_col] == val][feature_col].astype(str))
            wc_class = WordCloud(width=600, height=300, background_color="white").generate(subset_text)
            st.image(wc_class.to_array())
    except Exception as e:
        logging.error(f"Discourse Analysis Error: {e}")
        st.error("❌ Failed to generate class-wise word clouds.")

# ----- Target Distribution -----
with tabs[5]:
    st.markdown("### 🎯 Target Column Distribution")
    target_counts = df[target_col].value_counts().reset_index()
    target_counts.columns = [target_col, 'Count']
    fig_target = px.pie(target_counts, names=target_col, values='Count', hole=0.5)
    fig_target.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_target, use_container_width=True)

# -------------------------------
# Model Training Section
# -------------------------------
st.markdown("---")
st.subheader("🤖 Model Training & Recommendation")

X = df[feature_col].astype(str)
y = df[target_col]

# TF-IDF vectorization
try:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = tfidf.fit_transform(X)
except Exception as e:
    st.error("❌ Error during text vectorization. Please check your text column.")
    logging.error(f"TF-IDF Error: {e}")
    st.stop()

# Train-test split
try:
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        st.warning("⚠️ Some classes have <2 samples. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
except Exception as e:
    st.error("❌ Error during train-test split.")
    logging.error(f"Train-Test Split Error: {e}")
    st.stop()

# Apply SMOTE safely
try:
    if len(y_train.unique()) > 1:
        class_counts = Counter(y_train)
        min_class_size = min(class_counts.values())
        k_neighbors = max(1, min(5, min_class_size - 1))

        if min_class_size > 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            st.warning("⚠️ SMOTE skipped: Some classes have only one sample.")
    else:
        st.warning("⚠️ SMOTE skipped: Only one class present.")
except Exception as e:
    logging.error(f"SMOTE Error: {e}")
    st.error("❌ SMOTE balancing failed due to small class sizes or invalid data.")

# Define and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    except Exception as e:
        logging.error(f"Training failed for {name}: {e}")
        st.error(f"❌ Training failed for {name}. Check logs for details.")

# Display results
if results:
    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    st.dataframe(results_df)
    best_model = max(results, key=results.get)
    st.success(f"🏆 Recommended Model: **{best_model}** with Accuracy {results[best_model]*100:.2f}%")
else:
    st.warning("⚠️ No successful model training results available.")

# -------------------------------
# Predict User Input
# -------------------------------
st.markdown("### ✍️ Test Your Own Text")
user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip():
        try:
            user_vec = tfidf.transform([user_input])
            model = models.get(best_model)
            if model:
                prediction = model.predict(user_vec)[0]
                st.info(f"Prediction: **{prediction}**")
            else:
                st.warning("⚠️ No trained model found for prediction.")
        except Exception as e:
            logging.error(f"Prediction Error: {e}")
            st.error("❌ Prediction failed. Please ensure input is valid.")
    else:
        st.warning("Please enter some text to predict.")

# -------------------------------
# Optional: Download Error Logs
# -------------------------------
with open("app_errors.log", "r") as log_file:
    st.download_button("📜 Download Error Log", log_file, "error_log.txt")
