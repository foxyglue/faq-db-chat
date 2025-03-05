import os
import csv
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import getpass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-v1"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Data file paths
FAQ_CSV_PATH = "faq-data.csv"
DATABASE_JSON_PATH = "complaints.json"

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Set up embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

class RAGSystem:
    def __init__(self):
        self.faq_df = None
        self.faq_vectorstore = None
        self.db_vectorstore = None
        self.rag_graph = None
        self.source_info = None
        
    def load_faq_data(self, file_path: str) -> None:
        """Load FAQ data from CSV and create a vector store."""
        logger.info("Loading FAQ data from CSV...")
        self.faq_df = pd.read_csv(file_path)
        
        # Create documents for vector store
        documents = []
        for i, row in self.faq_df.iterrows():
            text = f"Question: {row['Questions']}\nFollowers: {row['Followers']}\nAnswered: {row['Answered']}\nLink: {row['Link']}"
            documents.append(Document(page_content=text, metadata={"source": "FAQ", "index": i}))
        
        # Create vector store
        self.faq_vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info(f"Loaded {len(documents)} FAQ entries into vector store")
    
    def load_database(self, file_path: str) -> None:
        """Load database from JSON and create a vector store."""
        logger.info("Loading database from JSON...")
        with open(file_path, 'r') as f:
            database = json.load(f)
        
        # Create documents for vector store
        documents = []
        for entry in database:
            source = entry["_source"]
            text = f"Complaint ID: {source['complaint_id']}\n"
            text += f"Issue: {source['issue']}\n"
            text += f"Sub-issue: {source['sub_issue']}\n"
            text += f"Product: {source['product']}\n"
            text += f"Sub-product: {source['sub_product']}\n"
            text += f"Company: {source['company']}\n"
            
            if source['complaint_what_happened']:
                text += f"Complaint Description: {source['complaint_what_happened']}\n"
            
            documents.append(Document(page_content=text, metadata={"source": "Database", "id": entry["_id"]}))
        
        # Create vector store
        self.db_vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info(f"Loaded {len(documents)} database entries into vector store")
    
    def search_faq(self, query: str, k: int = 3) -> Tuple[List[Document], bool]:
        """Search FAQ vectorstore for relevant documents using L2 distance scores.
        
        A document is considered relevant if its L2 distance is below the defined threshold.
        """
        logger.info(f"Searching FAQ data for: {query}")
        docs_with_scores = self.faq_vectorstore.similarity_search_with_scores(query, k=k)
        
        # Set the L2 distance threshold (lower scores indicate higher similarity)
        threshold = 0.3  
        filtered_docs = [doc for doc, score in docs_with_scores if score < threshold]
        
        # Consider the result relevant if at least one document meets the threshold
        relevant = len(filtered_docs) > 0
        return filtered_docs, relevant

    def search_database(self, query: str, k: int = 3) -> List[Document]:
        """Search database vectorstore for relevant documents."""
        logger.info(f"Searching database for: {query}")
        return self.db_vectorstore.similarity_search(query, k=k)
    
    def build_graph(self):
        """Build the LangGraph workflow."""
        
        # Define the state
        class State(dict):
            """The state of the RAG workflow."""
            query: str
            faq_docs: Optional[List[Document]] = None
            db_docs: Optional[List[Document]] = None
            source: Optional[str] = None
            response: Optional[str] = None

        # Define the nodes
        def query_router(state: State) -> str:
            """Route query to either FAQ or DB based on FAQ search results."""
            query = state["query"]
            faq_docs, relevant = self.search_faq(query)
            state["faq_docs"] = faq_docs
            
            if relevant:
                logger.info("Relevant FAQ documents found, using FAQ flow")
                return "faq_flow"
            else:
                logger.info("No relevant FAQ documents found, using database flow")
                return "db_flow"
        
        def faq_flow(state: State) -> State:
            """Generate a response from FAQ documents."""
            docs = state["faq_docs"]
            
            # Extract information from docs
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]
            
            # Create a prompt template for the FAQ
            prompt = ChatPromptTemplate.from_template("""
            You are an assistant tasked with answering questions based only on the provided FAQ information.
            
            FAQ Information:
            {context}
            
            Question: {query}
            
            Answer the question based only on the provided information. If the information doesn't contain a 
            clear answer, say so but provide the most relevant information you found.
            Include links when they are available.
            
            Important: Begin your response with "Source: FAQ" to clearly state where this information came from.
            """)
            
            # Create a chain
            chain = (
                {"context": lambda x: context, "query": lambda x: x["query"]}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(state)
            state["response"] = response
            state["source"] = "FAQ"
            
            # Store source info for later reference
            self.source_info = {
                "source": "FAQ",
                "documents": metadata
            }
            
            return state
        
        def db_flow(state: State) -> State:
            """Search database and generate a response."""
            query = state["query"]
            docs = self.search_database(query)
            state["db_docs"] = docs
            
            # Extract information from docs
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]
            
            # Create a prompt template for the database
            prompt = ChatPromptTemplate.from_template("""
            You are an assistant tasked with answering questions based only on the provided complaint database information.
            
            Database Information:
            {context}
            
            Question: {query}
            
            Answer the question based only on the provided information. If the information doesn't contain a
            clear answer, say so but provide the most relevant information you found.
            
            Important: Begin your response with "Source: Database" to clearly state where this information came from.
            """)
            
            # Create a chain
            chain = (
                {"context": lambda x: context, "query": lambda x: x["query"]}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(state)
            state["response"] = response
            state["source"] = "Database"
            
            # Store source info for later reference
            self.source_info = {
                "source": "Database",
                "documents": metadata
            }
            
            return state
        
        # Build the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("query_router", query_router)
        workflow.add_node("faq_flow", faq_flow)
        workflow.add_node("db_flow", db_flow)
        
        # Add edges
        workflow.add_conditional_edges(
            "query_router",
            {
                "faq_flow": lambda x: x == "faq_flow",
                "db_flow": lambda x: x == "db_flow"
            }
        )
        
        # Set the entrypoint
        workflow.set_entry_point("query_router")
        
        # Add final edges
        workflow.add_edge("faq_flow", END)
        workflow.add_edge("db_flow", END)
        
        # Compile the graph
        self.rag_graph = workflow.compile()
        
        return self.rag_graph
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """Get an answer for a query."""
        if not self.rag_graph:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        # Run the graph
        result = self.rag_graph.invoke({"query": query})
        
        # Return the answer and source information
        return {
            "query": query,
            "response": result["response"],
            "source": result["source"],
            "source_details": self.source_info
        }

# Usage example
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load data
    rag.load_faq_data(FAQ_CSV_PATH)
    rag.load_database(DATABASE_JSON_PATH)
    
    # Build the graph
    rag.build_graph()
    
    # Interactive loop for testing
    print("FAQ-Database RAG System initialized. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
            
        try:
            result = rag.get_answer(query)
            print(f"\n{result['response']}")
            print(f"\nSource Information: {result['source']}")
        except Exception as e:
            print(f"Error: {e}")