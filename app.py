import streamlit as st
from pypdf import PdfReader
import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

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

        # Combine retrieved chunks
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # -----------------------------
        # Prompt
        # -----------------------------
        prompt = f"""
You are a document QA assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

        # -----------------------------
        # LLM Call
        # -----------------------------
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # -----------------------------
        # Show Answer
        # -----------------------------
        st.subheader("AI Answer")
        st.write(answer)
