import os
import csv
import json
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google GenAI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def load_faq_data(csv_path):
    """
    Load FAQ data from CSV file
    """
    faqs = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            faqs.append({
                'question': row['Questions'],
                'followers': row['Followers'],
                'answered': row['Answered'],
                'link': row['Link']
            })
    return faqs

def load_complaint_data(json_path):
    """
    Load complaint data from JSON file
    """
    with open(json_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    
    complaints = []
    for item in data['root']:
        source = item.get('_source', {})
        complaint = {
            'complaint_id': source.get('complaint_id', ''),
            'issue': source.get('issue', ''),
            'product': source.get('product', ''),
            'sub_product': source.get('sub_product', ''),
            'company': source.get('company', ''),
            'complaint_what_happened': source.get('complaint_what_happened', '')
        }
        complaints.append(complaint)
    
    return complaints

def create_embeddings(texts, model_name='models/text-embedding-004'):
    """
    Create embeddings using Google GenAI
    """
    embedding_model = genai.embedding_models.get_model(model_name)
    embeddings = []
    
    for text in texts:
        # Combine all text fields into a single string for embedding
        full_text = ' '.join(str(v) for v in text.values() if v)
        embedding = embedding_model.embed_text(text=full_text)
        embeddings.append(embedding)
    
    return embeddings

def store_embeddings(faqs, complaints):
    """
    Store embeddings in ChromaDB
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_database")
    
    # Create collections
    faq_collection = chroma_client.create_collection("faq_collection")
    complaint_collection = chroma_client.create_collection("complaint_collection")
    
    # Embed and store FAQ data
    faq_texts = [
        {k: str(v) for k, v in faq.items()} for faq in faqs
    ]
    faq_embeddings = create_embeddings(faq_texts)
    
    faq_collection.add(
        embeddings=faq_embeddings,
        documents=[' '.join(str(v) for v in faq.values()) for faq in faq_texts],
        metadatas=faq_texts,
        ids=[f"faq_{i}" for i in range(len(faq_texts))]
    )
    
    # Embed and store Complaint data
    complaint_texts = [
        {k: str(v) for k, v in complaint.items()} for complaint in complaints
    ]
    complaint_embeddings = create_embeddings(complaint_texts)
    
    complaint_collection.add(
        embeddings=complaint_embeddings,
        documents=[' '.join(str(v) for v in complaint.values()) for complaint in complaint_texts],
        metadatas=complaint_texts,
        ids=[f"complaint_{i}" for i in range(len(complaint_texts))]
    )
    
    print(f"Stored {len(faq_texts)} FAQ embeddings")
    print(f"Stored {len(complaint_texts)} Complaint embeddings")

def main():
    # Paths to your data files
    faq_csv_path = 'faq-data.csv'
    complaint_json_path = 'complaintjson'
    
    # Load data
    faqs = load_faq_data(faq_csv_path)
    complaints = load_complaint_data(complaint_json_path)
    
    # Store embeddings
    store_embeddings(faqs, complaints)

if __name__ == "__main__":
    main()