# vectorize_and_save.py
import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from typing import List, Any

# Setup logging with debug level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def split_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a maximum length with a specified overlap.
    
    Args:
        text: The text to split.
        max_length: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.
        
    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end - overlap  # create overlap between chunks
    return chunks

def maybe_split_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Returns a list containing the text as a single chunk if its length is less than or equal 
    to max_length; otherwise, splits the text into chunks.
    """
    if len(text) <= max_length:
        logger.debug("Text length (%d) is under the threshold (%d). No splitting applied.", len(text), max_length)
        return [text]
    else:
        logger.debug("Text length (%d) exceeds threshold (%d). Splitting text.", len(text), max_length)
        return split_text(text, max_length, overlap)

class GenAIEmbeddings:
    """
    A wrapper for the Google Generative AI embedding API that also truncates text to a maximum byte size.
    """
    def __init__(self, model: str, task_type: str, max_bytes: int = 9000):
        self.model = model
        self.task_type = task_type
        self.max_bytes = max_bytes

    def truncate_text(self, text: str) -> str:
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > self.max_bytes:
            logger.debug("Text length (%d bytes) exceeds limit (%d bytes). Truncating text.", len(text_bytes), self.max_bytes)
            truncated = text_bytes[:self.max_bytes]
            return truncated.decode("utf-8", errors="ignore")
        return text

    def embed_query(self, text: str) -> List[float]:
        text = self.truncate_text(text)
        logger.debug("Embedding text (first 100 chars): %s", text[:100])
        result = genai.embed_content(
            model=self.model,
            task_type=self.task_type,
            content=text
        )
        return result["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def __call__(self, texts: Any) -> Any:
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)

# Initialize our embedding instance
embedding_model = "models/text-embedding-004"
embeddings = GenAIEmbeddings(model=embedding_model, task_type="RETRIEVAL_DOCUMENT")

# -------------------- Process FAQ Data --------------------
FAQ_CSV_PATH = "../faq-data.csv"
logger.info("Processing FAQ file: %s", FAQ_CSV_PATH)
faq_df = pd.read_csv(FAQ_CSV_PATH)
faq_documents = []
for i, row in faq_df.iterrows():
    text = (
        f"Question: {row['Questions']}\n"
        f"Followers: {row['Followers']}\n"
        f"Answered: {row['Answered']}\n"
        f"Link: {row['Link']}"
    )
    # Use conditional splitting: only split if text exceeds max_length.
    chunks = maybe_split_text(text, max_length=500, overlap=50)
    for j, chunk in enumerate(chunks):
        logger.debug("Processing FAQ document index: %d, chunk: %d", i, j)
        faq_documents.append(Document(page_content=chunk, metadata={"source": "FAQ", "index": i, "chunk": j}))

# Create FAQ vector store and save it
faq_vectorstore = FAISS.from_documents(faq_documents, embeddings)
faq_vectorstore.save_local("faq_vectorstore_index")
logger.info("FAQ vector store saved.")

# -------------------- Process Database Data --------------------
DATABASE_JSON_PATH = "../complaints.json"
logger.info("Processing Database file: %s", DATABASE_JSON_PATH)
with open(DATABASE_JSON_PATH, 'r') as f:
    database = json.load(f)
db_documents = []
for entry in database:
    source = entry["_source"]
    text = (
        f"Complaint ID: {source['complaint_id']}\n"
        f"Issue: {source['issue']}\n"
        f"Sub-issue: {source['sub_issue']}\n"
        f"Product: {source['product']}\n"
        f"Sub-product: {source['sub_product']}\n"
        f"Company: {source['company']}\n"
    )
    if source.get('complaint_what_happened'):
        text += f"Complaint Description: {source['complaint_what_happened']}\n"
    # Use conditional splitting: split only if text exceeds max_length.
    chunks = maybe_split_text(text, max_length=500, overlap=50)
    for j, chunk in enumerate(chunks):
        logger.debug("Processing Database document with ID: %s, chunk: %d", entry["_id"], j)
        db_documents.append(Document(page_content=chunk, metadata={"source": "Database", "id": entry["_id"], "chunk": j}))

# Create Database vector store and save it
db_vectorstore = FAISS.from_documents(db_documents, embeddings)
db_vectorstore.save_local("db_vectorstore_index")
logger.info("Database vector store saved.")

print("Vector stores saved!")
