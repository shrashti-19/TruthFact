import streamlit as st
import pickle
import re 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---- PAGE CONFIG ----
st.set_page_config(page_title="TruthFacT - AI Fact Checker", layout="wide")

# ---- TITLE ----
st.title("📰 TruthFacT - AI-Powered Fact-Checking")
st.write("Verify the authenticity of news and claims using AI.")

# ---- LOAD MODEL AND VECTORIZER ----
with open('pkl/logistic_model.pkl', "rb") as f:
    model = pickle.load(f)

with open("pkl/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---- PREPROCESS FUNCTION ----
def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# ---- USER INPUT ----
user_input = st.text_area("📝 Enter a news article or claim:", height=150)

# ---- PREDICTION ----
if st.button("🔎 Check Truth"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        st.info("🔍 Processing your input...")
        clean_input = preprocess(user_input)
        input_vector = vectorizer.transform([clean_input])
        
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = round(max(proba) * 100, 2)

        # Result Message
        if prediction == 1:
            st.error(f"🚨 This appears to be *Fake News*.")
        else:
            st.success(f"✅ This appears to be *True News*.")

        st.write(f"🧠 Prediction Confidence: `{confidence}%`")

        # Show pie chart
        fig, ax = plt.subplots()
        labels = ['True', 'Fake']
        ax.pie(proba, labels=labels, autopct='%1.1f%%', startangle=90, 
               colors=["#90ee90", "#ff6b6b"])
        ax.axis('equal')
        st.pyplot(fig)

# ---- SIDEBAR ----
st.sidebar.header("📌 About TruthFacT")
st.sidebar.markdown("""
**TruthFacT** is a lightweight, AI-powered tool built using:

- 🔤 TF-IDF Text Vectorization  
- 🔍 Logistic Regression Model  
- 🎨 Streamlit for Interface  

Designed to help you fact-check news articles and online claims in seconds!
""")
