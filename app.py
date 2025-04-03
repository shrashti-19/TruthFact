import streamlit as st

st.set_page_config(page_title="TruthFacT - AI Fact Checker", layout="wide")

st.title("📰 TruthFacT - AI-Powered Fact-Checking")
st.write("Verify the authenticity of news and claims using AI.")

user_input = st.text_area("Enter a news article or claim:", height=150)

if st.button("Check Truth"):
    st.write("🔍 Processing... (ML model will analyze this)")

st.sidebar.write("ℹ️ About TruthFacT")
st.sidebar.write("An AI-driven platform for fact-checking news and online claims.")