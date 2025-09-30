# ============================================
# 📌 Streamlit NLP Phase-wise with Model Comparison + SMOTE
# ============================================
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
from wordcloud import WordCloud

# -------------------------
# Load SpaCy model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="📰 Fake News Detection & NLP Analysis", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    h1, h2, h3, h4, h5 { color: #333333; }
    .css-1d391kg { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection & NLP Phase Analysis")
st.markdown("### Professional NLP Dashboard with SMOTE & Model Comparison")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Select Feature & Target
    # -------------------------
    feature_col = st.selectbox("Select Feature Column (Text Column)", df.columns)
    target_col = st.selectbox("Select Target Column", df.columns)

    st.markdown("---")
    st.subheader("📊 NLP Phase-wise Analysis")

    # ===== Tabs for NLP Phases =====
    tabs = st.tabs(["Lexical/Morphological", "Syntactic", "Semantic", "Pragmatic", "Discourse", "Target Distribution"])

    # ===== Lexical & Morphological =====
    with tabs[0]:
        st.markdown("### 1️⃣ Lexical & Morphological Analysis")
        df['word_count'] = df[feature_col].apply(lambda x: len(str(x).split()))
        df['char_count'] = df[feature_col].apply(lambda x: len(str(x)))

        col1, col2 = st.columns(2)
        fig1 = px.histogram(df, x='word_count', nbins=20, title="Word Count Distribution", color_discrete_sequence=['teal'])
        fig2 = px.histogram(df, x='char_count', nbins=20, title="Character Count Distribution", color_discrete_sequence=['orange'])
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

    # ===== Syntactic Analysis =====
    with tabs[1]:
        st.markdown("### 2️⃣ Syntactic Analysis (POS Tags)")
        def pos_tags(text):
            doc = nlp(str(text))
            return [token.pos_ for token in doc]
        df['pos_tags'] = df[feature_col].apply(pos_tags)
        pos_list = [tag for sublist in df['pos_tags'] for tag in sublist]
        pos_df = pd.Series(pos_list).value_counts().reset_index()
        pos_df.columns = ['POS', 'Count']
        fig_pos = px.bar(pos_df, x='POS', y='Count', title="POS Tag Distribution",
                         color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_pos, use_container_width=True)

    # ===== Semantic Analysis =====
    with tabs[2]:
        st.markdown("### 3️⃣ Semantic Analysis (WordCloud)")
        text = " ".join(df[feature_col].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(text)
        st.image(wc.to_array())

    # ===== Pragmatic Analysis =====
    with tabs[3]:
        st.markdown("### 4️⃣ Pragmatic Analysis (Sentence Length)")
        df['sentence_length'] = df[feature_col].apply(lambda x: len(str(x).split()))
        fig_prag = px.histogram(df, x='sentence_length', nbins=20, title="Sentence Length Distribution",
                                color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig_prag, use_container_width=True)

    # ===== Discourse Integration =====
    with tabs[4]:
        st.markdown("### 5️⃣ Discourse Integration (Class-wise WordClouds)")
        target_values = df[target_col].unique()
        for val in target_values:
            st.markdown(f"**Class: {val}**")
            subset_text = " ".join(df[df[target_col]==val][feature_col].astype(str))
            wc_class = WordCloud(width=600, height=300, background_color="white", colormap="tab10").generate(subset_text)
            st.image(wc_class.to_array())

    # ===== Target Distribution =====
    with tabs[5]:
        st.markdown("### 🎯 Target Column Distribution (Donut Chart)")
        target_counts = df[target_col].value_counts().reset_index()
        target_counts.columns = [target_col, 'Count']
        fig_target = px.pie(target_counts, names=target_col, values='Count', hole=0.5,
                            color=target_col, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_target.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_target, use_container_width=True)

    # -------------------------
    # Fake News Detection with SMOTE
    # -------------------------
    st.markdown("---")
    st.subheader("🤖 Fake News Detection Model (TF-IDF + SMOTE + Multiple Models)")

    # TF-IDF
    X = df[feature_col].astype(str)
    y = df[target_col]
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Multinomial NB": MultinomialNB()
    }

    model_scores = {}
    best_model_name = None
    best_acc = 0

    # Train & evaluate
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_scores[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    # Show accuracies
    score_df = pd.DataFrame(list(model_scores.items()), columns=["Model", "Accuracy"])
    fig_model = px.bar(score_df, x="Model", y="Accuracy", text="Accuracy",
                       color="Accuracy", color_continuous_scale="Viridis", title="Model Accuracy Comparison")
    st.plotly_chart(fig_model, use_container_width=True)

    st.success(f"✅ Best Model: **{best_model_name}** with Accuracy: {best_acc*100:.2f}%")

    # Predict user input
    st.markdown("### ✍️ Test Your Own Text")
    user_input = st.text_area("Enter news text here:")
    if st.button("Predict"):
        if user_input.strip() != "":
            user_vec = tfidf.transform([user_input])
            prediction = models[best_model_name].predict(user_vec)[0]
            st.info(f"Prediction: **{prediction}**")
        else:
            st.warning("Please enter some text to predict.")

else:
    st.info("Please upload a CSV file to get started.")
