import streamlit as st
from pypdf import PdfReader

st.title("📄 PDF Chatbot (Day 1)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.subheader("Extracted Text")
    st.text_area("PDF Content", text, height=300)