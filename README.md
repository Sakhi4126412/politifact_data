# LSSDP: NLP-Based Fake vs Real Statement Detection  

This project implements a **phase-wise NLP pipeline** to detect **Fake vs Real statements** using **Naive Bayes classifiers**.  
It follows the five major phases of Natural Language Processing (NLP):  
1. Lexical & Morphological Analysis  
2. Syntactic Analysis  
3. Semantic Analysis  
4. Discourse Integration  
5. Pragmatic Analysis  

Each phase extracts specific linguistic features and evaluates classification performance independently.  

---

## 🚀 Features
- **Dataset upload** (CSV with `Statement` and `BinaryTarget` columns)  
- **Preprocessing**: tokenization, lemmatization, stopword removal  
- **Syntactic analysis** using POS tagging (spaCy)  
- **Semantic analysis** with sentiment features (TextBlob)  
- **Discourse & pragmatic features** for deeper context  
- **Naive Bayes model** trained for each phase  
- **Phase-wise accuracy report** printed at the end  

---

## 📂 Project Structure
lssdp_streamlit.py # Main Python script
README.md # Project documentation
requirements.txt # Dependencies
