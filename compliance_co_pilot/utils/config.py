"""
Configuration management for the Compliance Co-Pilot module.

This module provides centralized configuration management for all
components of the compliance assessment system.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for LLM models used in compliance assessment."""
    
    # OpenAI configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="OpenAI model to use")
    openai_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Model temperature")
    openai_max_tokens: int = Field(4000, gt=0, description="Maximum tokens for response")
    
    # Alternative model configurations
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    anthropic_model: str = Field("claude-3-sonnet-20240229", description="Anthropic model to use")
    
    # Embedding model configuration
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model to use")
    embedding_dimensions: int = Field(1536, description="Embedding dimensions")
    
    # Model selection
    primary_model_provider: str = Field("openai", description="Primary model provider")
    
    @validator('openai_api_key', 'anthropic_api_key')
    def validate_api_keys(cls, v, values):
        """Validate that API keys are provided when needed."""
        if v is None:
            # Check if we can get from environment
            if 'openai_api_key' in values and values['openai_api_key'] is None:
                v = os.getenv('OPENAI_API_KEY')
            elif 'anthropic_api_key' in values and values['anthropic_api_key'] is None:
                v = os.getenv('ANTHROPIC_API_KEY')
        return v


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline components."""
    
    # Vector store configuration
    vector_store_type: str = Field("faiss", description="Type of vector store to use")
    vector_store_path: str = Field("./vector_store", description="Path to vector store")
    
    # Retrieval configuration
    retrieval_top_k: int = Field(5, gt=0, description="Number of documents to retrieve")
    retrieval_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold for retrieval")
    
    # Document processing
    chunk_size: int = Field(1000, gt=0, description="Size of document chunks")
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks")
    
    # Reranking configuration
    use_reranker: bool = Field(True, description="Whether to use document reranking")
    reranker_model: str = Field("ms-marco-MiniLM-L-12-v2", description="Reranker model to use")
    
    # Context window management
    max_context_length: int = Field(8000, gt=0, description="Maximum context length for LLM")
    context_strategy: str = Field("truncate", description="Strategy for handling long contexts")


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing components."""
    
    # Supported file types
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "xlsx", "txt"],
        description="Supported document formats"
    )
    
    # PDF processing
    pdf_extraction_method: str = Field("pymupdf", description="PDF extraction method")
    extract_tables: bool = Field(True, description="Whether to extract tables from documents")
    extract_images: bool = Field(False, description="Whether to extract images from documents")
    
    # Excel processing
    excel_sheet_names: Optional[List[str]] = Field(None, description="Specific Excel sheets to process")
    excel_skip_empty_rows: bool = Field(True, description="Skip empty rows in Excel")
    
    # Text processing
    text_encoding: str = Field("utf-8", description="Text encoding for documents")
    remove_headers_footers: bool = Field(True, description="Remove headers and footers")
    
    # Output configuration
    output_format: str = Field("markdown", description="Output format for processed documents")
    preserve_formatting: bool = Field(True, description="Preserve original formatting")


class ComplianceConfig(BaseModel):
    """Main configuration class for the Compliance Co-Pilot."""
    
    # Core configuration
    project_name: str = Field("AUgentik Compliance Co-Pilot", description="Project name")
    version: str = Field("0.1.0", description="Version number")
    
    # Model configuration
    models: ModelConfig = Field(default_factory=ModelConfig, description="Model configurations")
    
    # RAG configuration
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG pipeline configuration")
    
    # Document processing configuration
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig,
        description="Document processing configuration"
    )
    
    # File paths
    data_directory: str = Field("./data", description="Directory for data files")
    output_directory: str = Field("./output", description="Directory for output files")
    cache_directory: str = Field("./cache", description="Directory for cache files")
    
    # Processing configuration
    batch_size: int = Field(10, gt=0, description="Batch size for processing")
    max_workers: int = Field(4, gt=0, description="Maximum number of worker processes")
    timeout_seconds: int = Field(300, gt=0, description="Timeout for processing operations")
    
    # Logging configuration
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")
    
    # Assessment configuration
    assessment_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for assessments")
    max_assessment_retries: int = Field(3, gt=0, description="Maximum retries for failed assessments")
    
    # Output configuration
    generate_detailed_reports: bool = Field(True, description="Generate detailed compliance reports")
    include_evidence_citations: bool = Field(True, description="Include evidence citations in reports")
    export_formats: List[str] = Field(
        default=["json", "pdf", "excel"],
        description="Export formats for reports"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "COMPLIANCE_"
        case_sensitive = False
    
    def __init__(self, **data):
        """Initialize configuration with environment variable support."""
        # Load from environment variables if not provided
        for field_name, field_value in data.items():
            if field_value is None:
                env_var = f"COMPLIANCE_{field_name.upper()}"
                if env_var in os.environ:
                    data[field_name] = os.environ[env_var]
        
        super().__init__(**data)
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure that required directories exist."""
        directories = [
            self.data_directory,
            self.output_directory,
            self.cache_directory,
            self.rag.vector_store_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return self.models.dict()
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration as a dictionary."""
        return self.rag.dict()
    
    def get_document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration as a dictionary."""
        return self.document_processing.dict()
    
    def validate_configuration(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Check API keys
        if self.models.primary_model_provider == "openai" and not self.models.openai_api_key:
            issues.append("OpenAI API key is required when using OpenAI as primary provider")
        
        if self.models.primary_model_provider == "anthropic" and not self.models.anthropic_api_key:
            issues.append("Anthropic API key is required when using Anthropic as primary provider")
        
        # Check directories
        for directory in [self.data_directory, self.output_directory]:
            if not os.access(directory, os.W_OK):
                issues.append(f"Directory {directory} is not writable")
        
        # Check model parameters
        if self.models.openai_temperature < 0 or self.models.openai_temperature > 2:
            issues.append("OpenAI temperature must be between 0 and 2")
        
        if self.rag.retrieval_threshold < 0 or self.rag.retrieval_threshold > 1:
            issues.append("Retrieval threshold must be between 0 and 1")
        
        return issues


# Global configuration instance
config = ComplianceConfig()


def get_config() -> ComplianceConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> ComplianceConfig:
    """Update the global configuration with new values."""
    global config
    config = config.copy(update=kwargs)
    return config


def load_config_from_file(config_path: str) -> ComplianceConfig:
    """Load configuration from a JSON or YAML file."""
    import json
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return ComplianceConfig(**config_data) 