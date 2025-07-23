"""
RAG Pipeline for Compliance Assessment.

This module implements the Retrieval-Augmented Generation (RAG) pipeline
for compliance assessment, including document embedding, vector storage,
and intelligent retrieval of relevant context.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter

from ..utils.config import ComplianceConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline for compliance assessment.
    
    This class manages the complete RAG pipeline including document
    embedding, vector storage, retrieval, and context preparation
    for compliance assessment.
    """
    
    def __init__(self, config=None):
        """Initialize the RAG pipeline."""
        self.config = config or ComplianceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize vector store
        self.vector_store = None
        self.retriever = None
        
        # Initialize document compressors
        self.document_compressor = self._initialize_document_compressor()
        
        # Performance tracking
        self.retrieval_times = []
        self.embedding_times = []
    
    def _initialize_embeddings(self) -> Embeddings:
        """Initialize the embedding model."""
        try:
            if self.config.models.primary_model_provider == "openai":
                embeddings = OpenAIEmbeddings(
                    model=self.config.models.embedding_model,
                    dimensions=self.config.models.embedding_dimensions
                )
            else:
                # Fallback to OpenAI embeddings
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    dimensions=1536
                )
            
            self.logger.info(f"Initialized embeddings with model: {self.config.models.embedding_model}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def _initialize_document_compressor(self) -> Optional[DocumentCompressorPipeline]:
        """Initialize document compressor for reranking."""
        if not self.config.rag.use_reranker:
            return None
        
        try:
            # Create embeddings filter for reranking
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=self.config.rag.retrieval_threshold
            )
            
            # Create document compressor pipeline
            compressor_pipeline = DocumentCompressorPipeline(
                transformers=[embeddings_filter]
            )
            
            self.logger.info("Initialized document compressor for reranking")
            return compressor_pipeline
            
        except Exception as e:
            self.logger.warning(f"Could not initialize document compressor: {e}")
            return None
    
    def create_vector_store(self, documents: List[Document], store_name: str = "compliance_store") -> None:
        """
        Create and populate the vector store with documents.
        
        Args:
            documents: List of documents to embed and store
            store_name: Name for the vector store
        """
        self.logger.info(f"Creating vector store with {len(documents)} documents")
        
        start_time = time.time()
        
        try:
            # Create vector store
            if self.config.rag.vector_store_type.lower() == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # Save the vector store
                store_path = Path(self.config.rag.vector_store_path) / f"{store_name}.faiss"
                self.vector_store.save_local(str(store_path))
                
            elif self.config.rag.vector_store_type.lower() == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(Path(self.config.rag.vector_store_path) / store_name)
                )
            
            # Create retriever
            self.retriever = self._create_retriever()
            
            embedding_time = time.time() - start_time
            self.embedding_times.append(embedding_time)
            
            self.logger.info(f"Vector store created successfully in {embedding_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self, store_name: str = "compliance_store") -> None:
        """
        Load an existing vector store.
        
        Args:
            store_name: Name of the vector store to load
        """
        self.logger.info(f"Loading vector store: {store_name}")
        
        try:
            if self.config.rag.vector_store_type.lower() == "faiss":
                store_path = Path(self.config.rag.vector_store_path) / f"{store_name}.faiss"
                if store_path.exists():
                    self.vector_store = FAISS.load_local(
                        str(store_path),
                        embeddings=self.embeddings
                    )
                else:
                    raise FileNotFoundError(f"Vector store not found: {store_path}")
                    
            elif self.config.rag.vector_store_type.lower() == "chroma":
                store_path = Path(self.config.rag.vector_store_path) / store_name
                if store_path.exists():
                    self.vector_store = Chroma(
                        persist_directory=str(store_path),
                        embedding_function=self.embeddings
                    )
                else:
                    raise FileNotFoundError(f"Vector store not found: {store_path}")
            
            # Create retriever
            self.retriever = self._create_retriever()
            
            self.logger.info("Vector store loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise
    
    def _create_retriever(self) -> BaseRetriever:
        """Create the retriever with appropriate configuration."""
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before creating retriever")
        
        # Create base vector retriever
        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.rag.retrieval_top_k,
                "score_threshold": self.config.rag.retrieval_threshold
            }
        )
        
        # For now, use only vector retriever to avoid BM25 issues
        # TODO: Implement proper BM25 integration when needed
        
        # For now, return the vector retriever directly
        # TODO: Add document compression when needed
        self.logger.info("Created vector retriever")
        return vector_retriever
    
    def retrieve_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (overrides config)
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call create_vector_store() or load_vector_store() first.")
        
        start_time = time.time()
        
        try:
            # Use configured top_k if not specified
            if top_k is None:
                top_k = self.config.rag.retrieval_top_k
            
            # Retrieve documents
            documents = self.retriever.get_relevant_documents(query)
            
            # Limit to top_k if more documents were retrieved
            if len(documents) > top_k:
                documents = documents[:top_k]
            
            retrieval_time = time.time() - start_time
            self.retrieval_times.append(retrieval_time)
            
            self.logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f} seconds")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            raise
    
    def prepare_context_for_llm(self, documents: List[Document], query: str) -> str:
        """
        Prepare retrieved documents as context for the LLM.
        
        Args:
            documents: Retrieved documents
            query: Original query for context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        # Format documents for context
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Extract metadata for context
            metadata_info = []
            if doc.metadata:
                if "section" in doc.metadata:
                    metadata_info.append(f"Section: {doc.metadata['section']}")
                if "page_number" in doc.metadata:
                    metadata_info.append(f"Page: {doc.metadata['page_number']}")
                if "key_terms" in doc.metadata:
                    terms = doc.metadata["key_terms"][:3]  # Limit to first 3 terms
                    metadata_info.append(f"Key terms: {', '.join(terms)}")
            
            # Format document content
            doc_content = doc.page_content.strip()
            
            # Truncate if too long
            max_length = self.config.rag.max_context_length // len(documents)
            if len(doc_content) > max_length:
                doc_content = doc_content[:max_length] + "..."
            
            # Combine metadata and content
            if metadata_info:
                context_part = f"[Document {i} - {' | '.join(metadata_info)}]\n{doc_content}"
            else:
                context_part = f"[Document {i}]\n{doc_content}"
            
            context_parts.append(context_part)
        
        # Combine all context parts
        context = "\n\n".join(context_parts)
        
        # Add query context
        full_context = f"Query: {query}\n\nRelevant Documents:\n{context}"
        
        return full_context
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        stats = {
            "total_retrievals": len(self.retrieval_times),
            "total_embeddings": len(self.embedding_times),
            "average_retrieval_time": 0.0,
            "average_embedding_time": 0.0,
            "vector_store_type": self.config.rag.vector_store_type,
            "retrieval_top_k": self.config.rag.retrieval_top_k,
            "retrieval_threshold": self.config.rag.retrieval_threshold
        }
        
        if self.retrieval_times:
            stats["average_retrieval_time"] = sum(self.retrieval_times) / len(self.retrieval_times)
        
        if self.embedding_times:
            stats["average_embedding_time"] = sum(self.embedding_times) / len(self.embedding_times)
        
        return stats
    
    def add_documents_to_store(self, documents: List[Document]) -> None:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: New documents to add
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        self.logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Recreate retriever to include new documents
            self.retriever = self._create_retriever()
            
            self.logger.info("Documents added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def search_similar_documents(self, document: Document, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Find documents similar to a given document.
        
        Args:
            document: Document to find similar documents for
            top_k: Number of similar documents to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            # Get embedding for the document
            doc_embedding = self.embeddings.embed_query(document.page_content)
            
            # Search for similar documents
            similar_docs = self.vector_store.similarity_search_with_score(
                document.page_content,
                k=top_k
            )
            
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Error searching similar documents: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            # This is a simplified implementation
            # In practice, you might need to implement this based on your vector store
            all_docs = self.vector_store.get()
            
            if all_docs and 'documents' in all_docs:
                for doc in all_docs['documents']:
                    if doc.metadata.get('doc_id') == doc_id:
                        return doc
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving document by ID: {e}")
            return None
    
    def clear_vector_store(self) -> None:
        """Clear the vector store and reset retrievers."""
        self.vector_store = None
        self.retriever = None
        self.retrieval_times = []
        self.embedding_times = []
        
        self.logger.info("Vector store cleared")
    
    def export_vector_store_info(self) -> Dict[str, Any]:
        """Export information about the vector store."""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        try:
            info = {
                "vector_store_type": self.config.rag.vector_store_type,
                "embedding_model": self.config.models.embedding_model,
                "embedding_dimensions": self.config.models.embedding_dimensions,
                "retrieval_config": {
                    "top_k": self.config.rag.retrieval_top_k,
                    "threshold": self.config.rag.retrieval_threshold,
                    "use_reranker": self.config.rag.use_reranker
                },
                "performance_stats": self.get_retrieval_statistics()
            }
            
            # Try to get document count
            try:
                all_docs = self.vector_store.get()
                if all_docs and 'documents' in all_docs:
                    info["document_count"] = len(all_docs['documents'])
            except:
                info["document_count"] = "unknown"
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error exporting vector store info: {e}")
            return {"status": "error", "error": str(e)} 