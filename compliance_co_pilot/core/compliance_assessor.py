"""
Main Compliance Assessor for IFRS Compliance Analysis.

This module provides the main ComplianceAssessor class that orchestrates
the entire compliance assessment process, combining checklist parsing,
document processing, RAG retrieval, and LLM-based assessment.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from compliance_co_pilot.core.checklist_parser import ChecklistParser
from compliance_co_pilot.core.document_processor import DocumentProcessor
from compliance_co_pilot.core.rag_pipeline import RAGPipeline
from compliance_co_pilot.models.compliance_result import ComplianceResult, ComplianceStatus, EvidenceCitation, ComplianceReport
from compliance_co_pilot.models.ifrs_requirement import IFRSRequirement, IFRSStandard
from compliance_co_pilot.utils.config import ComplianceConfig

logger = logging.getLogger(__name__)


class TwoStepAssessmentOutput(BaseModel):
    """Structured output for the two-step assessment."""
    
    applicability_assessment: Dict[str, str] = Field(..., description="Applicability assessment results")
    compliance_assessment: Dict[str, Any] = Field(..., description="Compliance assessment results")

class ComplianceAssessmentOutput(BaseModel):
    """Structured output for compliance assessment."""
    
    compliance_status: str = Field(..., description="Compliance status: compliant, non_compliant, insufficient_info, not_applicable, requires_review")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for the compliance assessment")
    evidence_citations: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence found in AFS")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    relevant_afs_sections: List[str] = Field(default_factory=list, description="Relevant AFS sections reviewed")


class ComplianceAssessor:
    """
    Main compliance assessor for IFRS compliance analysis.
    
    This class orchestrates the complete compliance assessment process,
    from parsing checklists and processing documents to performing
    intelligent compliance assessments using RAG and LLMs.
    """
    
    def __init__(self, config=None):
        """Initialize the compliance assessor."""
        self.config = config or ComplianceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checklist_parser = ChecklistParser(config)
        self.document_processor = DocumentProcessor(config)
        self.rag_pipeline = RAGPipeline(config)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize assessment prompt
        self.assessment_prompt = self._create_assessment_prompt()
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=TwoStepAssessmentOutput)
        
        # Create assessment chain
        self.assessment_chain = self._create_assessment_chain()
        
        # Assessment tracking
        self.assessment_history = []
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM for compliance assessment."""
        try:
            if self.config.models.primary_model_provider == "openai":
                llm = ChatOpenAI(
                    model=self.config.models.openai_model,
                    temperature=self.config.models.openai_temperature,
                    max_tokens=self.config.models.openai_max_tokens
                )
            else:
                # Fallback to OpenAI
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=4000
                )
            
            self.logger.info(f"Initialized LLM: {self.config.models.openai_model}")
            return llm
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _create_assessment_prompt(self) -> ChatPromptTemplate:
        """Create the refined two-step assessment prompt template."""
        
        return ChatPromptTemplate.from_messages([
            ("system", """
You are an expert IFRS auditor with deep knowledge of International Financial Reporting Standards and their applicability across various industries. Your task is to perform a two-step assessment for a given IFRS requirement against an entity's financial statements:

**Step 1: Applicability Assessment**
First, determine if the specified IFRS standard/requirement is applicable to the entity based on its nature of business. Consider the core activities and industry of the entity. If the standard is clearly not relevant to the entity's operations (e.g., IFRS 17 for a manufacturing company), then it is NOT_APPLICABLE.

**Step 2: Compliance Assessment (ONLY if Applicable)**
If the standard/requirement is deemed APPLICABLE, then proceed to assess whether the provided Financial Statement Content complies with the specific IFRS requirement. Your assessment must be objective, evidence-based, and directly reference the provided text. If the Financial Statement Content does not provide sufficient information to make a definitive assessment, state 'INSUFFICIENT_INFO'.

Provide your assessment in a single, structured JSON format. Ensure all fields are present.
            """),
            ("human", """
Entity Name: {entity_name}
Entity Business Description: {entity_business_description}
IFRS Standard Name: {ifrs_standard_name}
IFRS Requirement: {ifrs_requirement}
Relevant Financial Statement Content: {afs_content}
Additional IFRS Standard Context (if available): {ifrs_standards_context}

Please provide your assessment in the following JSON format:
{{
    "applicability_assessment": {{
        "status": "[APPLICABLE | NOT_APPLICABLE]",
        "reasoning": "[Your detailed reasoning for the applicability assessment. Explain why the standard is or is not applicable to the entity's business.]"
    }},
    "compliance_assessment": {{
        "status": "[COMPLIANT | NON_COMPLIANT | INSUFFICIENT_INFO | N/A_NOT_APPLICABLE]",
        "reasoning": "[Your detailed reasoning for the compliance assessment. If 'N/A_NOT_APPLICABLE', state that compliance assessment was skipped due to non-applicability. Otherwise, explain how the AFS content addresses or fails to address the IFRS requirement. Be specific and refer to the provided text.]",
        "evidence_citations": [
            "[Direct quote or clear reference to the relevant part of the Financial Statement Content that supports your compliance assessment]",
            "[Another direct quote or reference, if applicable]"
        ],
        "suggested_follow_up": "[Optional: If INSUFFICIENT_INFO or NON_COMPLIANT, suggest what additional information or action would be needed to achieve compliance or complete the assessment. If N/A_NOT_APPLICABLE, state 'No follow-up required as standard is not applicable.']"
    }}
}}
            """)
        ])
    
    def _create_assessment_chain(self):
        """Create the assessment chain."""
        return (
            {
                "entity_name": RunnablePassthrough(),
                "entity_business_description": RunnablePassthrough(),
                "ifrs_standard_name": RunnablePassthrough(),
                "ifrs_requirement": RunnablePassthrough(),
                "afs_content": RunnablePassthrough(),
                "ifrs_standards_context": RunnablePassthrough()
            }
            | self.assessment_prompt
            | self.llm
            | self.output_parser
        )
    
    def assess_compliance(self, afs_file_path: str, checklist_file_path: str, 
                         company_name: str = "Unknown Company") -> ComplianceReport:
        """
        Perform comprehensive IFRS compliance assessment.
        
        Args:
            afs_file_path: Path to the Annual Financial Statement file
            checklist_file_path: Path to the IFRS compliance checklist file
            company_name: Name of the company being assessed
            
        Returns:
            ComplianceReport with comprehensive assessment results
        """
        self.logger.info(f"Starting compliance assessment for {company_name}")
        start_time = time.time()
        
        try:
            # Step 1: Parse IFRS checklist
            self.logger.info("Step 1: Parsing IFRS compliance checklist")
            ifrs_standard = self.checklist_parser.parse_checklist_file(checklist_file_path)
            
            # Validate checklist
            checklist_issues = self.checklist_parser.validate_checklist(checklist_file_path)
            if checklist_issues:
                self.logger.warning(f"Checklist validation issues: {checklist_issues}")
            
            # Step 2: Process AFS document
            self.logger.info("Step 2: Processing Annual Financial Statement")
            afs_data = self.document_processor.process_afs_file(afs_file_path)
            
            # Validate AFS content
            afs_issues = self.document_processor.validate_afs_content(afs_data)
            if afs_issues:
                self.logger.warning(f"AFS validation issues: {afs_issues}")
            
            # Step 3: Create RAG pipeline
            self.logger.info("Step 3: Setting up RAG pipeline")
            self.rag_pipeline.create_vector_store(afs_data["document_chunks"])
            
            # Step 4: Assess each requirement
            self.logger.info("Step 4: Assessing compliance requirements")
            compliance_results = []
            
            for requirement in ifrs_standard.all_requirements:
                try:
                    result = self._assess_single_requirement(requirement, afs_data, company_name, ifrs_standard)
                    compliance_results.append(result)
                    
                    # Log progress
                    if len(compliance_results) % 10 == 0:
                        self.logger.info(f"Assessed {len(compliance_results)}/{len(ifrs_standard.all_requirements)} requirements")
                        
                except Exception as e:
                    self.logger.error(f"Error assessing requirement {requirement.requirement_id}: {e}")
                    # Create error result
                    error_result = self._create_error_result(requirement, str(e))
                    compliance_results.append(error_result)
            
            # Step 5: Generate compliance report
            self.logger.info("Step 5: Generating compliance report")
            report = self._generate_compliance_report(
                compliance_results, afs_file_path, checklist_file_path, company_name
            )
            
            total_time = time.time() - start_time
            self.logger.info(f"Compliance assessment completed in {total_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in compliance assessment: {e}")
            raise
    
    def _assess_single_requirement(self, requirement: IFRSRequirement, 
                                  afs_data: Dict[str, Any], 
                                  company_name: str = "Unknown Company",
                                  ifrs_standard: IFRSStandard = None) -> ComplianceResult:
        """Assess compliance for a single IFRS requirement using two-step assessment."""
        start_time = time.time()
        
        # Create query for retrieval
        query = f"{requirement.requirement_text} {requirement.ifrs_reference}"
        
        # Retrieve relevant documents
        relevant_docs = self.rag_pipeline.retrieve_relevant_documents(query)
        
        # Prepare context
        afs_context = self.rag_pipeline.prepare_context_for_llm(relevant_docs, query)
        
        # Get IFRS standard context
        ifrs_context = self._get_ifrs_context(requirement)
        
        # Extract entity description from AFS data
        entity_description = self._extract_entity_description(afs_data)
        
        # Get IFRS standard name
        standard_name = ifrs_standard.name if ifrs_standard else f"IFRS Standard {requirement.ifrs_reference.split(':')[0]}"
        
        # Perform assessment
        try:
            assessment_output = self.assessment_chain.invoke({
                "entity_name": company_name,
                "entity_business_description": entity_description,
                "ifrs_standard_name": standard_name,
                "ifrs_requirement": requirement.requirement_text,
                "afs_content": afs_context,
                "ifrs_standards_context": ifrs_context
            })
            
            # Convert to ComplianceResult
            result = self._convert_two_step_assessment_to_result(
                assessment_output, requirement, relevant_docs, time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LLM assessment: {e}")
            return self._create_error_result(requirement, str(e))
    
    def _get_ifrs_context(self, requirement: IFRSRequirement) -> str:
        """Get relevant IFRS standard context for a requirement."""
        
        context_parts = [
            f"IFRS Standard: {requirement.ifrs_reference}",
            f"Category: {requirement.category}",
            f"Type: {requirement.requirement_type}",
            f"Mandatory: {'Yes' if requirement.mandatory else 'No'}",
            f"Requirement ID: {requirement.requirement_id}"
        ]
        
        # Add EY-specific guidance based on IFRS standard
        ey_guidance = self._get_ey_guidance(requirement.ifrs_reference)
        if ey_guidance:
            context_parts.append(f"EY Guidance: {ey_guidance}")
        
        if requirement.guidance_notes:
            context_parts.append(f"Additional Guidance: {requirement.guidance_notes}")
        
        # Add regulatory context
        regulatory_context = self._get_regulatory_context(requirement.ifrs_reference)
        if regulatory_context:
            context_parts.append(f"Regulatory Context: {regulatory_context}")
        
        return "\n".join(context_parts)
    
    def _get_ey_guidance(self, ifrs_reference: str) -> str:
        """Get EY-specific guidance for IFRS standard."""
        # This would ideally connect to the EY IFRS guidelines document
        # For now, providing key guidance based on common IFRS standards
        
        guidance_map = {
            "IFRS 1": "First-time adoption requires careful consideration of transition provisions and optional exemptions. Focus on materiality and consistency in application.",
            "IFRS 2": "Share-based payment transactions must be measured at fair value. Consider vesting conditions and market conditions carefully.",
            "IFRS 3": "Business combinations require purchase price allocation and goodwill calculation. Assess fair values of all identifiable assets and liabilities.",
            "IFRS 5": "Non-current assets held for sale must meet strict criteria. Measure at lower of carrying amount and fair value less costs to sell.",
            "IFRS 7": "Financial instruments disclosure requirements are extensive. Ensure all required quantitative and qualitative disclosures are provided.",
            "IFRS 8": "Operating segments must be identified based on internal reporting. Consider aggregation criteria and materiality thresholds.",
            "IFRS 9": "Financial instruments classification and measurement require careful analysis of business model and contractual cash flow characteristics.",
            "IFRS 10": "Consolidation requires assessment of control. Consider power, exposure to variable returns, and ability to affect returns.",
            "IFRS 11": "Joint arrangements require classification as joint operations or joint ventures. Assess rights and obligations carefully.",
            "IFRS 12": "Disclosure of interests in other entities must be comprehensive. Include all required quantitative and qualitative information.",
            "IFRS 13": "Fair value measurement requires use of appropriate valuation techniques. Consider market participant assumptions.",
            "IFRS 15": "Revenue recognition requires identification of performance obligations and allocation of transaction price. Consider variable consideration.",
            "IFRS 16": "Lease accounting requires recognition of right-of-use assets and lease liabilities. Assess lease term and discount rate carefully.",
            "IFRS 17": "Insurance contracts require complex measurement models. Consider risk adjustment and discounting requirements."
        }
        
        # Extract base IFRS standard (e.g., "IFRS 1" from "IFRS 1.39AH")
        base_standard = ifrs_reference.split('.')[0] if '.' in ifrs_reference else ifrs_reference
        
        return guidance_map.get(base_standard, "Standard IFRS application principles apply. Ensure compliance with recognition, measurement, and disclosure requirements.")
    
    def _get_regulatory_context(self, ifrs_reference: str) -> str:
        """Get regulatory context for IFRS standard."""
        # This would provide current regulatory expectations and enforcement trends
        
        regulatory_context = {
            "IFRS 1": "Regulators focus on proper application of transition provisions and consistency with future IFRS application.",
            "IFRS 2": "High regulatory scrutiny on share-based payment valuations and disclosure completeness.",
            "IFRS 3": "Regulators emphasize proper purchase price allocation and goodwill impairment testing.",
            "IFRS 5": "Strict enforcement of held-for-sale criteria and measurement requirements.",
            "IFRS 7": "Extensive disclosure requirements with regulatory focus on completeness and accuracy.",
            "IFRS 8": "Segment reporting receives significant regulatory attention for completeness and consistency.",
            "IFRS 9": "Expected credit loss models are closely monitored by regulators for proper application.",
            "IFRS 10": "Consolidation decisions are scrutinized for proper control assessment.",
            "IFRS 11": "Joint arrangement classification receives regulatory attention for proper application.",
            "IFRS 12": "Disclosure requirements are strictly enforced for completeness and transparency.",
            "IFRS 13": "Fair value measurements are closely monitored for proper valuation techniques and assumptions.",
            "IFRS 15": "Revenue recognition is a high-priority area for regulators with focus on proper application.",
            "IFRS 16": "Lease accounting implementation is monitored for proper recognition and measurement.",
            "IFRS 17": "Insurance contract accounting is under close regulatory scrutiny for proper implementation."
        }
        
        base_standard = ifrs_reference.split('.')[0] if '.' in ifrs_reference else ifrs_reference
        
        return regulatory_context.get(base_standard, "Standard regulatory expectations apply. Ensure proper application and adequate disclosure.")
    
    def _extract_entity_description(self, afs_data: Dict[str, Any]) -> str:
        """Extract entity business description from AFS data."""
        # Try to extract business description from various sources
        raw_text = afs_data.get("raw_text", "")
        
        # Look for common business description patterns
        patterns = [
            "nature of business",
            "business description",
            "company overview",
            "strategic report",
            "chief executive",
            "about us"
        ]
        
        # Find the first occurrence of any pattern and extract surrounding text
        for pattern in patterns:
            if pattern.lower() in raw_text.lower():
                start_idx = raw_text.lower().find(pattern.lower())
                end_idx = min(start_idx + 500, len(raw_text))  # Get 500 chars after pattern
                description = raw_text[start_idx:end_idx].strip()
                if len(description) > 50:  # Ensure we have meaningful content
                    return description
        
        # Fallback: return a generic description based on content analysis
        if "food" in raw_text.lower() or "ingredients" in raw_text.lower():
            return "Food and ingredients company"
        elif "manufacturing" in raw_text.lower():
            return "Manufacturing company"
        elif "financial" in raw_text.lower() or "bank" in raw_text.lower():
            return "Financial services company"
        else:
            return "Diversified business entity"
    
    def _convert_two_step_assessment_to_result(self, assessment_output: TwoStepAssessmentOutput, 
                                             requirement: IFRSRequirement, 
                                             relevant_docs: List, 
                                             processing_time: float) -> ComplianceResult:
        """Convert two-step assessment output to ComplianceResult."""
        
        # Extract applicability assessment
        applicability = assessment_output.applicability_assessment
        compliance = assessment_output.compliance_assessment
        
        # Determine final compliance status
        if applicability["status"] == "NOT_APPLICABLE":
            compliance_status = "not_applicable"
            reasoning = f"Standard not applicable: {applicability['reasoning']}"
            confidence_score = 0.9  # High confidence for non-applicability
            risk_level = "low"
        else:
            # Standard is applicable, use compliance assessment
            compliance_status = compliance["status"].lower().replace("n/a_not_applicable", "not_applicable")
            reasoning = f"Applicability: {applicability['reasoning']}\n\nCompliance: {compliance['reasoning']}"
            confidence_score = 0.7  # Moderate confidence for compliance assessment
            risk_level = "medium" if compliance_status == "non_compliant" else "low"
        
        # Convert evidence citations
        evidence_citations = []
        if "evidence_citations" in compliance:
            for citation in compliance["evidence_citations"]:
                if isinstance(citation, str):
                    # Handle string citations
                    evidence = EvidenceCitation(
                        source_section="Financial Statements",
                        page_number=None,
                        text_excerpt=citation,
                        confidence_score=0.5
                    )
                    evidence_citations.append(evidence)
        
        # Convert suggested actions
        suggested_actions = []
        if "suggested_follow_up" in compliance and compliance["suggested_follow_up"]:
            suggested_actions = [compliance["suggested_follow_up"]]
        
        # Create ComplianceResult
        result = ComplianceResult(
            requirement_id=requirement.requirement_id,
            requirement_text=requirement.requirement_text,
            ifrs_reference=requirement.ifrs_reference,
            compliance_status=compliance_status,
            confidence_score=confidence_score,
            reasoning=reasoning,
            evidence_citations=evidence_citations,
            suggested_actions=suggested_actions,
            risk_level=risk_level,
            model_used=self.config.models.openai_model,
            processing_time_seconds=processing_time,
            relevant_afs_sections=["Financial Statements"]  # Default section
        )
        
        return result
    
    def _create_error_result(self, requirement: IFRSRequirement, error_message: str) -> ComplianceResult:
        """Create a ComplianceResult for an assessment error."""
        return ComplianceResult(
            requirement_id=requirement.requirement_id,
            requirement_text=requirement.requirement_text,
            ifrs_reference=requirement.ifrs_reference,
            compliance_status=ComplianceStatus.INSUFFICIENT_INFO,
            confidence_score=0.0,
            reasoning=f"Assessment failed due to error: {error_message}",
            evidence_citations=[],
            suggested_actions=["Review assessment process", "Check system configuration"],
            risk_level="high",
            model_used=self.config.models.openai_model,
            processing_time_seconds=0.0,
            relevant_afs_sections=[]
        )
    
    def _generate_compliance_report(self, compliance_results: List[ComplianceResult],
                                  afs_file_path: str, checklist_file_path: str,
                                  company_name: str) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        
        # Calculate statistics
        total_requirements = len(compliance_results)
        compliant_count = len([r for r in compliance_results if r.compliance_status == ComplianceStatus.COMPLIANT])
        non_compliant_count = len([r for r in compliance_results if r.compliance_status == ComplianceStatus.NON_COMPLIANT])
        insufficient_info_count = len([r for r in compliance_results if r.compliance_status == ComplianceStatus.INSUFFICIENT_INFO])
        
        # Calculate overall compliance score
        if total_requirements > 0:
            overall_compliance_score = compliant_count / total_requirements
        else:
            overall_compliance_score = 0.0
        
        # Identify critical issues
        critical_issues = []
        high_priority_actions = []
        
        for result in compliance_results:
            if result.risk_level == "critical" and result.compliance_status != ComplianceStatus.COMPLIANT:
                critical_issues.append(f"{result.ifrs_reference}: {result.requirement_text[:100]}...")
            
            if result.risk_level in ["high", "critical"] and result.suggested_actions:
                high_priority_actions.extend(result.suggested_actions[:2])  # Limit to 2 actions per requirement
        
        # Remove duplicates
        high_priority_actions = list(set(high_priority_actions))
        
        # Create report
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            company_name=company_name,
            assessment_date=datetime.now(),
            total_requirements=total_requirements,
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            insufficient_info_count=insufficient_info_count,
            overall_compliance_score=overall_compliance_score,
            critical_issues=critical_issues,
            high_priority_actions=high_priority_actions,
            compliance_results=compliance_results,
            afs_file_path=afs_file_path,
            checklist_file_path=checklist_file_path,
            model_configuration={"model": self.config.models.openai_model}
        )
        
        return report
    
    def export_report(self, report: ComplianceReport, output_path: str, format: str = "json"):
        """Export compliance report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(report.dict(), f, indent=2, default=str)
        elif format.lower() == "excel":
            self._export_report_to_excel(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Report exported to {output_path}")
    
    def _export_report_to_excel(self, report: ComplianceReport, output_path: Path):
        """Export report to Excel format."""
        import pandas as pd
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                "Metric": [
                    "Total Requirements",
                    "Compliant",
                    "Non-Compliant", 
                    "Insufficient Info",
                    "Overall Compliance Score",
                    "Critical Issues",
                    "High Priority Actions"
                ],
                "Value": [
                    report.total_requirements,
                    report.compliant_count,
                    report.non_compliant_count,
                    report.insufficient_info_count,
                    f"{report.overall_compliance_score:.2%}",
                    len(report.critical_issues),
                    len(report.high_priority_actions)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Detailed results sheet
            results_data = []
            for result in report.compliance_results:
                results_data.append({
                    "Requirement ID": result.requirement_id,
                    "IFRS Reference": result.ifrs_reference,
                    "Requirement Text": result.requirement_text[:200] + "..." if len(result.requirement_text) > 200 else result.requirement_text,
                    "Compliance Status": result.compliance_status.value,
                    "Confidence Score": result.confidence_score,
                    "Risk Level": result.risk_level,
                    "Reasoning": result.reasoning[:300] + "..." if len(result.reasoning) > 300 else result.reasoning,
                    "Suggested Actions": "; ".join(result.suggested_actions[:3])
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
            
            # Critical issues sheet
            if report.critical_issues:
                issues_df = pd.DataFrame({
                    "Critical Issue": report.critical_issues
                })
                issues_df.to_excel(writer, sheet_name="Critical Issues", index=False)
            
            # High priority actions sheet
            if report.high_priority_actions:
                actions_df = pd.DataFrame({
                    "High Priority Action": report.high_priority_actions
                })
                actions_df.to_excel(writer, sheet_name="High Priority Actions", index=False)
    
    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get statistics about the assessment process."""
        return {
            "total_assessments": len(self.assessment_history),
            "rag_statistics": self.rag_pipeline.get_retrieval_statistics(),
            "model_configuration": self.config.get_model_config(),
            "last_assessment": self.assessment_history[-1] if self.assessment_history else None
        } 