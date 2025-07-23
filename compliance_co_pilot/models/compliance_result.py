"""
Data models for compliance assessment results.

This module defines the core data structures used to represent
compliance assessment results, including status enums and result objects.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ComplianceStatus(str, Enum):
    """Enumeration of possible compliance statuses."""
    
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_INFO = "insufficient_info"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class EvidenceCitation(BaseModel):
    """Represents a citation to evidence found in the financial statements."""
    
    source_section: str = Field(..., description="Section of the AFS where evidence was found")
    page_number: Optional[int] = Field(None, description="Page number in the AFS")
    text_excerpt: str = Field(..., description="Relevant text excerpt from the AFS")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in this evidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ComplianceResult(BaseModel):
    """Represents the result of a compliance assessment for a single IFRS requirement."""
    
    requirement_id: str = Field(..., description="Unique identifier for the IFRS requirement")
    requirement_text: str = Field(..., description="Text description of the IFRS requirement")
    ifrs_reference: str = Field(..., description="IFRS standard reference (e.g., IFRS 15.1)")
    
    # Assessment results
    compliance_status: ComplianceStatus = Field(..., description="Overall compliance status")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment")
    
    # Detailed analysis
    reasoning: str = Field(..., description="Detailed reasoning for the compliance assessment")
    evidence_citations: List[EvidenceCitation] = Field(default_factory=list, description="Evidence found in AFS")
    
    # Action items
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    
    # Metadata
    assessment_timestamp: datetime = Field(default_factory=datetime.now, description="When assessment was performed")
    model_used: str = Field(..., description="LLM model used for assessment")
    processing_time_seconds: float = Field(..., description="Time taken for assessment")
    
    # Additional context
    relevant_afs_sections: List[str] = Field(default_factory=list, description="Relevant AFS sections reviewed")
    ifrs_standard_context: Optional[str] = Field(None, description="Relevant IFRS standard context")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComplianceReport(BaseModel):
    """Aggregated compliance report for all assessed requirements."""
    
    report_id: str = Field(..., description="Unique identifier for the compliance report")
    company_name: str = Field(..., description="Name of the company being assessed")
    assessment_date: datetime = Field(default_factory=datetime.now, description="Date of assessment")
    
    # Summary statistics
    total_requirements: int = Field(..., description="Total number of requirements assessed")
    compliant_count: int = Field(..., description="Number of compliant requirements")
    non_compliant_count: int = Field(..., description="Number of non-compliant requirements")
    insufficient_info_count: int = Field(..., description="Number with insufficient information")
    
    # Overall assessment
    overall_compliance_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance percentage")
    critical_issues: List[str] = Field(default_factory=list, description="Critical compliance issues found")
    high_priority_actions: List[str] = Field(default_factory=list, description="High priority actions required")
    
    # Detailed results
    compliance_results: List[ComplianceResult] = Field(..., description="Individual compliance results")
    
    # Metadata
    afs_file_path: str = Field(..., description="Path to the Annual Financial Statement file")
    checklist_file_path: str = Field(..., description="Path to the IFRS checklist file")
    model_configuration: Dict[str, Any] = Field(default_factory=dict, description="Model configuration used")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the compliance report."""
        return {
            "total_requirements": self.total_requirements,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "insufficient_info_count": self.insufficient_info_count,
            "overall_compliance_score": self.overall_compliance_score,
            "critical_issues_count": len(self.critical_issues),
            "high_priority_actions_count": len(self.high_priority_actions),
        }
    
    def get_results_by_status(self, status: ComplianceStatus) -> List[ComplianceResult]:
        """Get all results for a specific compliance status."""
        return [result for result in self.compliance_results if result.compliance_status == status]
    
    def get_high_risk_results(self) -> List[ComplianceResult]:
        """Get all results with high or critical risk levels."""
        return [
            result for result in self.compliance_results 
            if result.risk_level in ["high", "critical"]
        ] 