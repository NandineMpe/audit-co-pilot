"""
Document Processor for Annual Financial Statements.

This module handles the processing and structuring of Annual Financial
Statements (AFS) from various formats, primarily PDF, to prepare them
for compliance assessment.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders.base import BaseLoader

from ..utils.config import ComplianceConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processor for Annual Financial Statements and related documents.
    
    This class handles the extraction, cleaning, and structuring of
    financial documents to prepare them for compliance assessment.
    """
    
    def __init__(self, config=None):
        """Initialize the document processor."""
        self.config = config or ComplianceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Common financial statement sections
        self.financial_sections = {
            "balance_sheet": [
                "balance sheet", "statement of financial position", "financial position",
                "assets", "liabilities", "equity", "shareholders' equity"
            ],
            "income_statement": [
                "income statement", "profit and loss", "statement of comprehensive income",
                "revenue", "expenses", "net income", "profit", "loss"
            ],
            "cash_flow": [
                "cash flow", "statement of cash flows", "cash flows",
                "operating activities", "investing activities", "financing activities"
            ],
            "notes": [
                "notes to financial statements", "accounting policies",
                "significant accounting policies", "notes and explanations"
            ],
            "directors_report": [
                "directors' report", "management discussion", "board report",
                "management's discussion and analysis", "md&a"
            ],
            "audit_report": [
                "auditor's report", "independent auditor's report", "audit opinion",
                "report of independent auditors"
            ]
        }
    
    def process_afs_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an Annual Financial Statement file.
        
        Args:
            file_path: Path to the AFS file (PDF, DOCX, etc.)
            
        Returns:
            Dictionary containing processed document sections and metadata
        """
        self.logger.info(f"Processing AFS file: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AFS file not found: {file_path}")
        
        # Load the document
        documents = self._load_document(file_path)
        
        # Extract and structure content
        structured_content = self._extract_structured_content(documents)
        
        # Process and clean text
        processed_sections = self._process_sections(structured_content)
        
        # Create document chunks for RAG
        document_chunks = self._create_document_chunks(processed_sections)
        
        # Generate metadata
        metadata = self._generate_metadata(file_path, structured_content)
        
        result = {
            "file_path": file_path,
            "metadata": metadata,
            "sections": processed_sections,
            "document_chunks": document_chunks,
            "raw_text": structured_content["raw_text"],
            "tables": structured_content["tables"],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Successfully processed AFS file with {len(document_chunks)} chunks")
        return result
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate loader based on file type."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredFileLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} document pages")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            raise
    
    def _extract_structured_content(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract structured content from documents."""
        structured_content = {
            "raw_text": "",
            "pages": [],
            "tables": [],
            "sections": {},
            "metadata": {}
        }
        
        # Combine all pages
        for i, doc in enumerate(documents):
            page_content = doc.page_content
            structured_content["raw_text"] += f"\n\n--- Page {i+1} ---\n\n{page_content}"
            structured_content["pages"].append({
                "page_number": i + 1,
                "content": page_content,
                "metadata": doc.metadata
            })
        
        # Extract sections
        structured_content["sections"] = self._identify_sections(structured_content["raw_text"])
        
        # Extract tables (basic extraction - could be enhanced)
        structured_content["tables"] = self._extract_tables(structured_content["raw_text"])
        
        return structured_content
    
    def _identify_sections(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Identify and extract different sections of the financial statements."""
        sections = {}
        text_lower = text.lower()
        
        for section_name, keywords in self.financial_sections.items():
            section_info = self._find_section(text, text_lower, keywords)
            if section_info:
                sections[section_name] = section_info
        
        return sections
    
    def _find_section(self, text: str, text_lower: str, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Find a specific section in the text based on keywords."""
        for keyword in keywords:
            # Look for section headers
            pattern = rf"(?i)({keyword}[^.]*?)(?:\n|\.|$)"
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                start_pos = match.start()
                
                # Extract section content (next 2000 characters or until next section)
                end_pos = min(start_pos + 2000, len(text))
                section_content = text[start_pos:end_pos]
                
                # Find the actual section boundaries
                section_boundaries = self._find_section_boundaries(text, start_pos)
                
                return {
                    "keyword_found": keyword,
                    "start_position": start_pos,
                    "end_position": section_boundaries["end"],
                    "content": text[section_boundaries["start"]:section_boundaries["end"]],
                    "confidence": 0.8
                }
        
        return None
    
    def _find_section_boundaries(self, text: str, start_pos: int) -> Dict[str, int]:
        """Find the boundaries of a section starting from a given position."""
        # Look for common section separators
        separators = [
            r"\n\s*\d+\.\s*",  # Numbered sections
            r"\n\s*[A-Z][A-Z\s]+\n",  # ALL CAPS headers
            r"\n\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\n",  # Title Case headers
            r"\n\s*[-=]{3,}\n",  # Separator lines
        ]
        
        end_pos = len(text)
        for separator in separators:
            matches = re.finditer(separator, text[start_pos + 100:])  # Skip first 100 chars
            for match in matches:
                potential_end = start_pos + 100 + match.start()
                if potential_end > start_pos + 500:  # Minimum section length
                    end_pos = min(end_pos, potential_end)
                    break
        
        return {
            "start": start_pos,
            "end": end_pos
        }
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from the text (basic implementation)."""
        tables = []
        
        # Look for table-like patterns
        table_patterns = [
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # Numbers in columns
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # Text + numbers
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                table_info = {
                    "start_position": match.start(),
                    "end_position": match.end(),
                    "content": match.group(0),
                    "type": "basic_table"
                }
                tables.append(table_info)
        
        return tables
    
    def _process_sections(self, structured_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean the extracted sections."""
        processed_sections = {}
        
        for section_name, section_info in structured_content["sections"].items():
            if section_info:
                processed_content = self._clean_section_content(section_info["content"])
                processed_sections[section_name] = {
                    **section_info,
                    "processed_content": processed_content,
                    "word_count": len(processed_content.split()),
                    "key_terms": self._extract_key_terms(processed_content)
                }
        
        return processed_sections
    
    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers and headers
        content = re.sub(r'Page \d+', '', content)
        content = re.sub(r'\d+\s*of\s*\d+', '', content)
        
        # Remove common PDF artifacts
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', content)
        
        # Normalize spacing around punctuation
        content = re.sub(r'\s+([.,;:!?])', r'\1', content)
        
        return content.strip()
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key financial terms from content."""
        # Common financial terms
        financial_terms = [
            "revenue", "expenses", "assets", "liabilities", "equity",
            "cash", "debt", "income", "profit", "loss", "depreciation",
            "amortization", "goodwill", "intangible", "inventory",
            "accounts receivable", "accounts payable", "retained earnings"
        ]
        
        content_lower = content.lower()
        found_terms = []
        
        for term in financial_terms:
            if term in content_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _create_document_chunks(self, processed_sections: Dict[str, Any]) -> List[Document]:
        """Create document chunks for RAG processing."""
        chunks = []
        
        for section_name, section_info in processed_sections.items():
            if "processed_content" in section_info:
                # Split section content into chunks
                section_chunks = self.text_splitter.split_text(section_info["processed_content"])
                
                for i, chunk_text in enumerate(section_chunks):
                    # Create metadata for the chunk
                    metadata = {
                        "section": section_name,
                        "chunk_index": i,
                        "total_chunks": len(section_chunks),
                        "word_count": len(chunk_text.split()),
                        "key_terms": section_info.get("key_terms", []),
                        "confidence": section_info.get("confidence", 0.8),
                        "source_type": "afs_section"
                    }
                    
                    # Create Document object
                    doc = Document(
                        page_content=chunk_text,
                        metadata=metadata
                    )
                    chunks.append(doc)
        
        return chunks
    
    def _generate_metadata(self, file_path: str, structured_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for the processed document."""
        file_path_obj = Path(file_path)
        
        metadata = {
            "file_name": file_path_obj.name,
            "file_size": file_path_obj.stat().st_size,
            "file_extension": file_path_obj.suffix,
            "processing_date": datetime.now().isoformat(),
            "total_pages": len(structured_content["pages"]),
            "sections_found": list(structured_content["sections"].keys()),
            "tables_found": len(structured_content["tables"]),
            "total_words": len(structured_content["raw_text"].split()),
            "document_type": "annual_financial_statement"
        }
        
        return metadata
    
    def extract_company_info(self, processed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract company information from the AFS."""
        company_info = {
            "name": None,
            "fiscal_year": None,
            "reporting_currency": None,
            "industry": None
        }
        
        # Extract company name
        company_name_patterns = [
            r"Annual Report\s+of\s+([A-Z][A-Za-z\s&.,]+?)(?:\s+for|\s+year|\n)",
            r"([A-Z][A-Za-z\s&.,]+?)\s+Annual Report",
            r"Financial Statements\s+of\s+([A-Z][A-Za-z\s&.,]+?)"
        ]
        
        raw_text = processed_content.get("raw_text", "")
        for pattern in company_name_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                company_info["name"] = match.group(1).strip()
                break
        
        # Extract fiscal year
        year_patterns = [
            r"for the year ended\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            r"year ended\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            r"(\d{4})\s+Annual Report"
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                company_info["fiscal_year"] = match.group(1)
                break
        
        # Extract reporting currency
        currency_patterns = [
            r"in\s+([A-Z]{3})\s+millions?",
            r"expressed in\s+([A-Z]{3})",
            r"currency:\s+([A-Z]{3})"
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                company_info["reporting_currency"] = match.group(1)
                break
        
        return company_info
    
    def validate_afs_content(self, processed_content: Dict[str, Any]) -> List[str]:
        """Validate the processed AFS content and return any issues."""
        issues = []
        
        # Check if essential sections are present
        essential_sections = ["balance_sheet", "income_statement", "notes"]
        missing_sections = []
        
        for section in essential_sections:
            if section not in processed_content["sections"]:
                missing_sections.append(section)
        
        if missing_sections:
            issues.append(f"Missing essential sections: {missing_sections}")
        
        # Check content quality
        total_words = len(processed_content["raw_text"].split())
        if total_words < 1000:
            issues.append(f"Document appears too short: {total_words} words")
        
        # Check for common issues
        if "error" in processed_content["raw_text"].lower():
            issues.append("Document contains error messages")
        
        if len(processed_content["tables"]) == 0:
            issues.append("No tables found in document")
        
        return issues 