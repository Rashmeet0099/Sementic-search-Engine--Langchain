import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings

# -------------------------------------------
# STEP 1: Load environment and Gemini API key
# -------------------------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------------------------
# STEP 2: Document Loader — Load PDF into Pages
# -------------------------------------------------
pdf_path = "sample.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Filter out blank or irrelevant pages
pages = [page for page in pages if len(page.page_content.strip()) > 100]

# ------------------------------------------------------------
# STEP 3: Text Chunking — Split long documents into segments
# ------------------------------------------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

# Optional: Remove chunks with spam/unwanted content
documents = [
    doc for doc in documents
    if "Capgemini" not in doc.page_content[:300] and len(doc.page_content.strip()) > 100
]

# --------------------------------------------------------
# STEP 4: Embedding — Define cosine-normalized wrapper
# --------------------------------------------------------
class NormalizedEmbeddings(Embeddings):
    def __init__(self, base_embeddings):
        self.base = base_embeddings

    def embed_documents(self, texts):
        return [self._normalize(v) for v in self.base.embed_documents(texts)]

    def embed_query(self, text):
        return self._normalize(self.base.embed_query(text))

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

# Load Gemini Embedding Model
raw_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
normalized_embeddings = NormalizedEmbeddings(raw_embeddings)

# --------------------------------------------------------------
# STEP 5: Vectorization — Create FAISS index with document vectors
# --------------------------------------------------------------
if not documents:
    raise ValueError("No valid documents to index. Check filters or input PDF content.")

vectorstore = FAISS.from_documents(documents, normalized_embeddings)

# -------------------------------------------------------------
# STEP 6: Query & Cosine Similarity — Embed + Search top chunks
# -------------------------------------------------------------
query = "What is LangChain?"
query_embedding = normalized_embeddings.embed_query(query)

results = vectorstore.similarity_search(query, k=6)

# -------------------------------------------------------------
# STEP 7: Output — Top matching results with their embeddings
# -------------------------------------------------------------
print("\n🔎 Top 3 Most Relevant Results with Cosine Embeddings:\n")
for i, doc in enumerate(results[:3]):
    print(f"🔹 {i+1}. Content:\n{doc.page_content.strip()}\n")
    
    doc_embedding = normalized_embeddings.embed_query(doc.page_content)
    print(f"📐 Embedding (first 10 values): {doc_embedding[:10]} ... [Total dims: {len(doc_embedding)}]\n")
