🤖 PDF Chatbot (RAG-Based)

An AI-powered PDF Question Answering application built using Retrieval-Augmented Generation (RAG).

Users can upload PDF documents and ask natural language questions about the content. The application retrieves relevant information using semantic search and generates grounded answers using an LLM.

🚀 Features


📄 Upload PDF documents
🧠 Semantic search using embeddings
🔍 Retrieval-Augmented Generation (RAG)
💬 ChatGPT-style chat interface
⚡ Fast similarity search with FAISS
🤖 LLM-powered answers using Llama 3
🎯 Context-grounded responses


🛠 Tech Stack
Technology	Purpose
Python	Backend logic
Streamlit	Frontend UI
PyPDF	PDF text extraction
LangChain	Text chunking
HuggingFace	Embedding generation
FAISS	Vector database
Groq API	LLM inference
Llama 3	AI response generation


⚙️ Installation
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/pdf-chatbot.git

3. Move Into Project Folder
cd pdf-chatbot

4. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Mac/Linux
python3 -m venv venv
source venv/bin/activate
📦 Install Dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root folder:

GROQ_API_KEY=your_api_key_here

Get your free API key from:

https://console.groq.com

▶️ Run Application

streamlit run app.py

📌 Key AI Concepts Used
Retrieval-Augmented Generation (RAG)
Embeddings
Semantic Search
Vector Databases
Prompt Engineering
LLM Integration
Context Grounding
💡 Future Improvements
Multi-PDF support
OCR for scanned PDFs
Streaming responses
Persistent vector database
Authentication
Chat history memory
Deployment to cloud
🧑‍💻 Author

Bilal Shahzad

