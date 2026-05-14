# =========================================
# PDF CHATBOT (RAG-Based)
# =========================================

# ---------- IMPORTS ----------

# Streamlit for frontend UI
import streamlit as st

# PDF text extraction
from pypdf import PdfReader

# Text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database
from langchain_community.vectorstores import FAISS

# Environment variables
import os
from dotenv import load_dotenv

# Groq client (OpenAI-compatible)
from openai import OpenAI


# ---------- LOAD ENV VARIABLES ----------

load_dotenv()

# Get API key from .env
api_key = os.getenv("GROQ_API_KEY")


# ---------- INITIALIZE GROQ CLIENT ----------

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ---------- STREAMLIT PAGE CONFIG ----------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ---------- SIDEBAR ----------

st.sidebar.title("📄 PDF Chatbot")
st.sidebar.write("RAG-Based Document QA System")

st.sidebar.markdown("---")

st.sidebar.write("### Tech Stack")
st.sidebar.write("- Streamlit")
st.sidebar.write("- FAISS")
st.sidebar.write("- HuggingFace Embeddings")
st.sidebar.write("- Groq + Llama 3")


# ---------- MAIN TITLE ----------

st.title("🤖 AI PDF Chatbot")
st.caption("Ask questions about your PDF using RAG")


# ---------- PDF UPLOAD ----------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


# ---------- PROCESS PDF ----------

if uploaded_file:

    # Extract text from PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        # Prevent NoneType errors
        if extracted:
            text += extracted

   

    # ---------- CHUNKING ----------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # Sidebar metrics
    st.sidebar.write(f"### Total Chunks: {len(chunks)}")

    # ---------- EMBEDDINGS ----------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------- VECTOR STORE ----------

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # Save vector store in session state
    st.session_state.vector_store = vector_store

   
    # ---------- CHAT INPUT ----------

    question = st.chat_input(
        "Ask a question about the PDF"
    )

    # Run only if question exists
    if question:

        # ---------- USER MESSAGE ----------

        with st.chat_message("user"):
            st.write(question)

        # ---------- RETRIEVAL ----------

        docs = st.session_state.vector_store.similarity_search(
            question,
            k=3
        )

        # Combine retrieved chunks
        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        # ---------- PROMPT ----------

        prompt = f"""
You are a document QA assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know."

Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

        # ---------- GENERATE RESPONSE ----------

        with st.spinner("Thinking..."):

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

        # ---------- ASSISTANT MESSAGE ----------

        with st.chat_message("assistant"):
            st.write(answer)

        

       
