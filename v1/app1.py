# application.py

import os
import json
import logging
import getpass
from dotenv import load_dotenv
from typing import List, Tuple, Any
import pandas as pd
import google.generativeai as genai

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define our embedding wrapper class that uses google.generativeai
class GenAIEmbeddings:
    def __init__(self, model: str, task_type: str):
        self.model = model
        self.task_type = task_type

    def embed_query(self, text: str) -> List[float]:
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

# Initialize the LLM (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Define the RAG system class
class RAGSystem:
    def __init__(self):
        self.faq_vectorstore = None
        self.db_vectorstore = None
        self.rag_graph = None
        self.source_info = None

    def load_vectorstores(self) -> None:
        """Load precomputed vector stores from disk."""
        try:
            self.faq_vectorstore = FAISS.load_local("faq_vectorstore_index", embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded FAQ vectorstore from disk.")
        except Exception as e:
            logger.error("Failed to load FAQ vectorstore: " + str(e))

        try:
            self.db_vectorstore = FAISS.load_local("db_vectorstore_index", embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded Database vectorstore from disk.")
        except Exception as e:
            logger.error("Failed to load Database vectorstore: " + str(e))

    def search_faq(self, query: str, k: int = 3) -> Tuple[List[Document], bool]:
        """Search FAQ vectorstore for relevant documents."""
        logger.info(f"Searching FAQ data for: {query}")
        # Use simple similarity search instead of with_scores
        docs = self.faq_vectorstore.similarity_search(query, k=k)
        # Assume documents are relevant if any are returned
        relevant = len(docs) > 0
        return docs, relevant

    def search_database(self, query: str, k: int = 3) -> List[Document]:
        """Search database vectorstore for relevant documents."""
        logger.info(f"Searching database for: {query}")
        return self.db_vectorstore.similarity_search(query, k=k)

    def build_graph(self):
        """Build the RAG workflow."""
        # Define the state
        class State(dict):
            query: str
            faq_docs: List[Document] = None
            db_docs: List[Document] = None
            source: str = None
            response: str = None

        # Define the nodes
        def query_router(state: State) -> str:
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
            docs = state["faq_docs"]
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]

            # Create a prompt template for FAQ flow
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are an assistant tasked with answering questions based only on the provided FAQ information.
                
                FAQ Information:
                {context}
                
                Question: {query}
                
                Answer the question based only on the provided information. If the information doesn't contain a 
                clear answer, say so but provide the most relevant information you found.
                Include links when they are available.
                
                Important: Begin your response with "Source: FAQ" to clearly state where this information came from.
                """
            )

            prompt = prompt_template.format(context=context, query=state["query"])

            # Call the LLM with the prompt to generate an answer
            response = llm.invoke(prompt)
            state["response"] = response
            state["source"] = "FAQ"
            self.source_info = {"source": "FAQ", "documents": metadata}
            return state

        def db_flow(state: State) -> State:
            query = state["query"]
            docs = self.search_database(query)
            state["db_docs"] = docs
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]

            # Create a prompt template for Database flow
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are an assistant tasked with answering questions based only on the provided complaint database information.
                
                Database Information:
                {context}
                
                Question: {query}                
                
                Answer the question based only on the provided information. If the information doesn't contain a
                clear answer, say so but provide the most relevant information you found.
                
                Important: Begin your response with "Source: Database" to clearly state where this information came from.
                """
            )

            prompt = prompt_template.format(context=context, query=query)

            # Call the LLM with the prompt to generate an answer
            response = llm.invoke(prompt)
            state["response"] = response
            state["source"] = "Database"
            self.source_info = {"source": "Database", "documents": metadata}
            return state

        # Build a simple workflow graph (dictionary-based)
        self.rag_graph = {
            "query_router": query_router,
            "faq_flow": faq_flow,
            "db_flow": db_flow
        }

        def invoke(state: dict) -> dict:
            next_node = self.rag_graph["query_router"](state)

            if next_node == "faq_flow":
                state = self.rag_graph["faq_flow"](state)
            elif next_node == "db_flow":
                state = self.rag_graph["db_flow"](state)
            return state

        self.rag_graph["invoke"] = invoke

    def get_answer(self, query: str) -> dict:
        """Get an answer for a query using the built workflow."""
        
        if not self.rag_graph or "invoke" not in self.rag_graph:
            raise ValueError("Graph not built. Call build_graph() first.")
        result = self.rag_graph["invoke"]({"query": query})

        return {
            "query": query,
            "response": result["response"],
            "source": result["source"],
            "source_details": self.source_info
        }


if __name__ == "__main__":
    # Initialize RAG system and load vector stores
    rag = RAGSystem()
    rag.load_vectorstores()
    rag.build_graph()

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