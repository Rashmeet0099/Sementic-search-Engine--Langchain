# Sementic-search-Engine--Langchain

# 🔍 Semantic Search Engine using LangChain, FAISS & Google Gemini

This project is a **Semantic Search Engine** built using LangChain, FAISS, and Google Gemini embeddings. It loads content from a PDF file (`sample.pdf`), splits it into overlapping text chunks, filters irrelevant content, and generates cosine-normalized embeddings using Gemini's `models/embedding-001`. These embeddings are stored in a FAISS vector index to enable fast and accurate semantic search. When a user submits a query (e.g., “What is LangChain?”), the system retrieves the most relevant document chunks based on vector similarity. This setup is ideal for building document-based AI search tools and RAG (Retrieval-Augmented Generation) pipelines.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/semantic-search-engine.git
cd semantic-search-engine
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Add Gemini API Key
Create a .env file in the root folder and add your Google Gemini API key:

env
Copy
Edit
GOOGLE_API_KEY=your_google_api_key_here
👉 Get your Gemini API Key here

🚀 Run the Script
Ensure you have your sample.pdf file ready in the project folder, then run:

bash
Copy
Edit
python main.py
📄 Example Output
sql
Copy
Edit
🔎 Top 3 Most Relevant Results with Cosine Embeddings:

🔹 1. Content:
LangChain is an open-source framework designed to simplify...

📐 Embedding (first 10 values): [0.123, 0.345, ..., 0.087] ... [Total dims: 768]
📦 Project Structure
bash
Copy
Edit
semantic-search-engine/
├── sample.pdf           # The PDF file to search from
├── main.py              # Main Python script
├── .env                 # API key environment file
├── requirements.txt     # List of dependencies
└── README.md            # This readme file
✅ Features
PDF document loader

Intelligent text chunking

Gemini-powered embeddings with cosine normalization

FAISS vector store indexing

Semantic query search with top-k relevant results

📚 Dependencies
langchain

langchain-community

langchain-google-genai

faiss-cpu

numpy

python-dotenv

PyPDF2 (used by PyPDFLoader)

You can install them using:

bash
Copy
Edit
pip install -r requirements.txt
