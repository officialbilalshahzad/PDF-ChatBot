import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("📄 PDF Chatbot - Day 3")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # -----------------------------
    # Extract Text from PDF
    # -----------------------------
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    st.success("PDF text extracted successfully!")

    # -----------------------------
    # Split into Chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.write(f"Total Chunks: {len(chunks)}")

    # -----------------------------
    # Create Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Store in FAISS
    # -----------------------------
    vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("Vector store created!")

    # -----------------------------
    # User Question
    # -----------------------------
    question = st.text_input("Ask a question about the PDF")

    if question:

        # Search similar chunks
        docs = vector_store.similarity_search(
            question,
            k=3
        )

        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(docs):

            st.write(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.write("-------------------")
