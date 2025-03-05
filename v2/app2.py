import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configure Google GenAI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class GraphState(TypedDict):
    """
    Represents the state of our graph during RAG
    """
    question: str
    faq_context: List[dict]
    complaint_context: List[dict]
    answer: Optional[str]
    source: Optional[str]

def retrieve_faq_context(state: GraphState):
    """
    Retrieve relevant FAQ context based on the question
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_database")
    faq_collection = chroma_client.get_collection("faq_collection")
    
    # Embed the query
    embedding_model = genai.embedding_models.get_model('models/text-embedding-004')
    query_embedding = embedding_model.embed_text(text=state['question'])
    
    # Retrieve top 3 most similar FAQs
    results = faq_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    # Transform results into context
    faq_context = [
        {
            'question': meta['question'], 
            'followers': meta['followers'], 
            'answered': meta['answered'], 
            'link': meta['link']
        } for meta in results['metadatas'][0]
    ]
    
    return {**state, 'faq_context': faq_context}

def retrieve_complaint_context(state: GraphState):
    """
    Retrieve relevant complaint context based on the question
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_database")
    complaint_collection = chroma_client.get_collection("complaint_collection")
    
    # Embed the query
    embedding_model = genai.embedding_models.get_model('models/text-embedding-004')
    query_embedding = embedding_model.embed_text(text=state['question'])
    
    # Retrieve top 3 most similar complaints
    results = complaint_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    # Transform results into context
    complaint_context = [
        {
            'complaint_id': meta['complaint_id'], 
            'issue': meta['issue'], 
            'product': meta['product'], 
            'sub_product': meta['sub_product'], 
            'company': meta['company'],
            'complaint_what_happened': meta['complaint_what_happened']
        } for meta in results['metadatas'][0]
    ]
    
    return {**state, 'complaint_context': complaint_context}

def generate_answer(state: GraphState):
    """
    Generate an answer using retrieved context and GenAI
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare context
    faq_context = "\n".join([
        f"FAQ Question: {ctx['question']}, Followers: {ctx['followers']}, Link: {ctx['link']}" 
        for ctx in state['faq_context']
    ])
    
    complaint_context = "\n".join([
        f"Complaint ID: {ctx['complaint_id']}, Issue: {ctx['issue']}, Product: {ctx['product']}, Company: {ctx['company']}" 
        for ctx in state['complaint_context']
    ])
    
    # Construct prompt
    system_prompt = f"""
    You are a helpful AI assistant. Answer the user's question based strictly on the provided context.
    
    FAQ Context:
    {faq_context}
    
    Complaint Context:
    {complaint_context}
    
    If no relevant information is found in the context, respond with "I could not find a precise answer in the available data."
    """
    
    # Generate response
    response = model.generate_content(
        system_prompt + f"\n\nUser Question: {state['question']}"
    )
    
    # Determine source
    source = "FAQ" if state['faq_context'] else "Complaints" if state['complaint_context'] else "No source"
    
    return {
        **state, 
        'answer': response.text,
        'source': source
    }

def build_rag_graph():
    """
    Build the LangGraph for RAG workflow
    """
    workflow = StateGraph(GraphState)
    
    # Define graph nodes
    workflow.add_node("retrieve_faq", retrieve_faq_context)
    workflow.add_node("retrieve_complaint", retrieve_complaint_context)
    workflow.add_node("generate_answer", generate_answer)
    
    # Define graph edges
    workflow.set_entry_point("retrieve_faq")
    workflow.add_edge("retrieve_faq", "retrieve_complaint")
    workflow.add_edge("retrieve_complaint", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

def main():
    # Initialize the RAG graph
    rag_graph = build_rag_graph()
    
    while True:
        # Get user input
        question = input("\nAsk a question (or type 'exit' to quit): ")
        
        if question.lower() == 'exit':
            break
        
        # Run the RAG workflow
        result = rag_graph.invoke({"question": question})
        
        # Print the answer and its source
        print("\nAnswer:", result['answer'])
        print("Source:", result['source'])

if __name__ == "__main__":
    main()