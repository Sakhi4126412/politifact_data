import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
from wordcloud import WordCloud

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Fake News Detection & NLP Analysis", layout="wide")
st.title("📰 Fake News Detection & NLP Phase Analysis")

# Upload CSV
st.markdown("<h5 style='text-align: center; color: grey;'>Upload your dataset (CSV)</h5>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ⚡ Clean missing values in selected columns
    df.dropna(inplace=True)

    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select feature & target columns
    feature_col = st.selectbox("Select Feature Column (Text Column)", df.columns)
    target_col = st.selectbox("Select Target Column", df.columns)

    # ⚡ Ensure selected columns are valid
    if feature_col and target_col:
        df = df[[feature_col, target_col]].dropna()
        df[feature_col] = df[feature_col].astype(str)

        st.markdown("---")
        st.subheader("NLP Phase-wise Analysis")

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

            fig_pos = px.bar(pos_df, x='POS', y='Count', title="POS Tag Distribution", color='Count', color_continuous_scale='Viridis')
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
            fig_prag = px.histogram(df, x='sentence_length', nbins=20, title="Sentence Length Distribution", color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig_prag, use_container_width=True)

        # ===== Discourse Integration =====
        with tabs[4]:
            st.markdown("### 5️⃣ Discourse Integration (Class-wise WordClouds)")
            target_values = df[target_col].unique()
            for val in target_values:
                st.markdown(f"**Class: {val}**")
                subset_text = " ".join(df[df[target_col]==val][feature_col].astype(str))
                if subset_text.strip():
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

        # ===== Fake News Detection Model =====
        st.markdown("---")
        st.subheader("🤖 Fake News Detection Model (TF-IDF + Logistic Regression)")

        # Train model
        X = df[feature_col].astype(str)
        y = df[target_col]
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X_tfidf = tfidf.fit_transform(X)

        # ⚡ Safe stratify option
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            stratify_option = None
        else:
            stratify_option = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=stratify_option
        )

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model Trained Successfully! Accuracy: {acc*100:.2f}%")

        # Classification Report
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        st.dataframe(report_df)

        # Predict user input
        st.markdown("### ✍️ Test Your Own Text")
        user_input = st.text_area("Enter news text here:")
        if st.button("Predict"):
            if user_input.strip() != "":
                user_vec = tfidf.transform([user_input])
                prediction = model.predict(user_vec)[0]
                st.info(f"Prediction: **{prediction}**")
            else:
                st.warning("Please enter some text to predict.")

else:
    st.info("Please upload a CSV file to get started.")
