import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports to fix deprecation warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import mlx.core as mx
from mlx_lm import load, generate

class MLX_RAG_System:
    def __init__(self, persistence_dir: str = "./chroma_db", model_path: str = "~/mlx-models"):
        """Initialize the RAG system with MLX LLM and a persistent ChromaDB directory."""
        self.persistence_dir = persistence_dir
        self.model_path = os.path.expanduser(model_path)  # Expand the ~ in the path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.current_pdf_pages = {}
        self.model = None
        self.tokenizer = None
        
        # Initialize vector DB
        self.initialize_vector_db()
        
        # Keep track of processed documents
        self.processed_docs = set()
        
        # Load the model at initialization
        self.load_mlx_model()

    def initialize_vector_db(self):
        """Initialize the vector database with persistence."""
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
        
        self.vector_db = Chroma(
            persist_directory=self.persistence_dir,
            embedding_function=self.embeddings
        )
    
    def load_mlx_model(self):
        """Load the MLX model if not already loaded."""
        if self.model is None:
            with st.spinner("Loading MLX model (this may take a moment)..."):
                self.model, self.tokenizer = load(self.model_path)
                st.success("Model loaded successfully!")

    def process_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Process a PDF file and extract text by page."""
        reader = PdfReader(pdf_path)
        pages_dict = {}
        
        # Create a progress bar
        total_pages = len(reader.pages)
        progress_text = f"Processing PDF ({total_pages} pages)"
        my_bar = st.progress(0, text=progress_text)
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                pages_dict[page_num] = page_text.strip()
            # Update progress bar
            my_bar.progress(page_num / total_pages, text=f"{progress_text}: Page {page_num}/{total_pages}")
        
        # Clear the progress bar when done
        my_bar.empty()
        
        self.current_pdf_pages = pages_dict
        return pages_dict

    def process_text_for_storage(self, text: str, doc_id: str, chunk_size: int = 1000, 
                                chunk_overlap: int = 200) -> None:
        """Process any text into chunks and store in vector database."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_text(text)
        chunks_metadata = []
        
        for chunk_num, _ in enumerate(chunks):
            chunks_metadata.append({
                "chunk": chunk_num + 1,
                "doc_id": doc_id,
                "source_type": "text"
            })

        self.vector_db.add_texts(
            texts=chunks,
            metadatas=chunks_metadata
        )
        self.processed_docs.add(doc_id)

    def process_pages_for_storage(self, pages_dict: Dict[int, str], chunk_size: int = 1000, 
                                 chunk_overlap: int = 200) -> Tuple[List[str], List[Dict]]:
        """Process pages into chunks for vector storage."""
        chunks = []
        chunks_metadata = []
        
        for page_num, page_text in pages_dict.items():
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk_num, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                chunks_metadata.append({
                    "page": page_num,
                    "chunk": chunk_num + 1,
                    "total_pages": len(pages_dict),
                    "source_type": "pdf"
                })

        return chunks, chunks_metadata

    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]], doc_id: str):
        """Add document chunks to the vector database with metadata."""
        for meta in metadata:
            meta["doc_id"] = doc_id

        self.vector_db.add_texts(
            texts=chunks,
            metadatas=metadata
        )
        # In newer versions of langchain_chroma, persist() is no longer needed as it's automatic
        # The collection is persisted after each add_texts() call
        self.processed_docs.add(doc_id)

    def query_with_mlx_stream(self, prompt: str, response_placeholder, max_tokens: int = 5000) -> str:
        """
        Generate a response using the MLX model with streaming output.
        Shows intermediate results in the Streamlit UI.
        """
        # Make sure model is loaded
        if self.model is None:
            self.load_mlx_model()
        
        # Generate response with streaming
        full_response = ""
        
        # Generate initial tokens
        output = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens
        )
        
        # Update the placeholder with the processed response
        response_placeholder.markdown(output)
        
        return output

    def query_system(self, user_query: str, response_placeholder, 
                    filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the system with user input, optionally with filters.
        Streams the response to the UI as it's being generated.
        
        Args:
            user_query: User's question
            response_placeholder: Streamlit placeholder for showing streaming response
            filter_dict: Optional dictionary for filtering vector search results
        """
        # Retrieve relevant chunks based on query
        if filter_dict:
            results = self.vector_db.similarity_search(user_query, k=4, filter=filter_dict)
        else:
            results = self.vector_db.similarity_search(user_query, k=4)
        
        # Extract context and source information
        context_chunks = []
        source_info = []
        
        for doc in results:
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            chunk_text = doc.page_content
            
            if source_type == "pdf":
                page_num = metadata.get("page", "unknown")
                context_chunks.append(f"[Page {page_num}]: {chunk_text}")
                source_info.append({
                    "page": page_num,
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "source_type": "pdf"
                })
            elif source_type == "youtube":
                #timestamp = metadata.get("timestamp", "unknown")
                context_chunks.append(f"[Video Transcript]: {chunk_text}")
                source_info.append({
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "source_type": "youtube"
                })
            else:
                context_chunks.append(chunk_text)
                source_info.append({
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "source_type": source_type
                })

        context = "\n\n".join(context_chunks)
        
        # Prepare prompt with source awareness and explicit reasoning instructions
        prompt = f"""Based on the following context, please answer the question.
First, think through your reasoning process using <think> and </think> tags, then provide your final answer.

<think>
When analyzing this question, carefully examine the provided context.
Identify relevant information and connect the dots step by step.
Consider what information is most important to answer the question accurately.
</think>

If you reference information from the text, mention the source if available.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {user_query}

"""

        # Generate streaming response using MLX
        response_text = self.query_with_mlx_stream(prompt, response_placeholder, max_tokens=5000)
        
        return {
            "response": response_text,
            "sources": source_info
        }

    def reset_system(self):
        """Reset the system by creating a new ChromaDB instance at a new location."""
        try:
            # Close existing connection if possible
            if hasattr(self.vector_db, '_client') and self.vector_db._client is not None:
                try:
                    self.vector_db._client = None
                except:
                    pass
            
            # Import needed libraries
            import os
            import time
            
            # Create a new persistence directory with timestamp
            base_dir = os.path.dirname(self.persistence_dir) or "."
            new_dir = os.path.join(base_dir, f"chroma_db_{int(time.time())}")
            os.makedirs(new_dir, exist_ok=True)
            
            # Update the persistence directory
            self.persistence_dir = new_dir
            
            # Create a fresh embeddings object
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create a completely new vector DB instance with the new path
            self.vector_db = Chroma(
                persist_directory=new_dir,
                embedding_function=self.embeddings
            )
            
            # Clear processed docs set
            self.processed_docs.clear()
            
            # Clear current PDF pages
            self.current_pdf_pages = {}
            
            return True, f"System reset successfully! New database created at {new_dir}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return False, f"Error resetting system: {e}"