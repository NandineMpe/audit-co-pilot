"""
AUgentik Compliance Co-Pilot Module

A sophisticated IFRS compliance assessment system built on LangChain that automates
the review of Annual Financial Statements against IFRS standards.

This module provides:
- Automated IFRS compliance checklist ingestion and processing
- Financial statement analysis and document processing
- Intelligent compliance assessment using RAG and LLMs
- Structured output with reasoning and evidence citations
- Interactive reporting and visualization capabilities
"""

from .core.compliance_assessor import ComplianceAssessor
from .core.document_processor import DocumentProcessor
from .core.checklist_parser import ChecklistParser
from .core.rag_pipeline import RAGPipeline
from .models.compliance_result import ComplianceResult, ComplianceStatus
from .models.ifrs_requirement import IFRSRequirement
from .utils.config import ComplianceConfig

__version__ = "0.1.0"
__author__ = "AUgentik Team"

__all__ = [
    "ComplianceAssessor",
    "DocumentProcessor", 
    "ChecklistParser",
    "RAGPipeline",
    "ComplianceResult",
    "ComplianceStatus",
    "IFRSRequirement",
    "ComplianceConfig",
] 