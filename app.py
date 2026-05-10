import streamlit as st
from pypdf import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("📄 PDF Chatbot ")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.subheader("Raw Text Preview")
    st.text_area("PDF Text", text[:2000], height=200)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.subheader("Chunks Created")
    st.write(f"Total Chunks: {len(chunks)}")

    st.write(chunks[0])


    # Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert chunks into vectors
vector = embeddings.embed_query(chunks[0])

st.subheader("Embedding Preview")

st.write(vector[:10])

# Create FAISS vector store
vector_store = FAISS.from_texts(chunks, embeddings)

st.success("FAISS vector store created successfully!")
