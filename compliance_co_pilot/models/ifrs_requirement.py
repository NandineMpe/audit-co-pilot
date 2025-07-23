"""
Data models for IFRS requirements.

This module defines the data structures used to represent
IFRS compliance requirements extracted from checklists.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class IFRSRequirement(BaseModel):
    """Represents a single IFRS compliance requirement."""
    
    requirement_id: str = Field(..., description="Unique identifier for the requirement")
    requirement_text: str = Field(..., description="Text description of the requirement")
    ifrs_reference: str = Field(..., description="IFRS standard reference (e.g., IFRS 15.1)")
    
    # Hierarchical structure
    category: str = Field(..., description="High-level category (e.g., Revenue Recognition)")
    subcategory: Optional[str] = Field(None, description="Subcategory within the main category")
    section: Optional[str] = Field(None, description="Specific section within the standard")
    
    # Requirement details
    requirement_type: str = Field(..., description="Type of requirement: disclosure, measurement, presentation")
    mandatory: bool = Field(..., description="Whether this requirement is mandatory")
    materiality_threshold: Optional[str] = Field(None, description="Materiality considerations")
    
    # Additional context
    guidance_notes: Optional[str] = Field(None, description="Additional guidance or notes")
    examples: List[str] = Field(default_factory=list, description="Example implementations")
    related_requirements: List[str] = Field(default_factory=list, description="Related requirement IDs")
    
    # Metadata
    checklist_source: str = Field(..., description="Source checklist file")
    version: str = Field(..., description="IFRS standard version")
    effective_date: Optional[str] = Field(None, description="Effective date of the requirement")
    
    # Processing metadata
    extraction_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in requirement extraction")
    processing_notes: List[str] = Field(default_factory=list, description="Notes from processing")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_full_reference(self) -> str:
        """Get the full IFRS reference including section details."""
        reference_parts = [self.ifrs_reference]
        if self.section:
            reference_parts.append(self.section)
        return " ".join(reference_parts)
    
    def get_hierarchical_path(self) -> str:
        """Get the hierarchical path for this requirement."""
        path_parts = [self.category]
        if self.subcategory:
            path_parts.append(self.subcategory)
        if self.section:
            path_parts.append(self.section)
        return " > ".join(path_parts)
    
    def is_related_to(self, other_requirement: "IFRSRequirement") -> bool:
        """Check if this requirement is related to another requirement."""
        return (
            other_requirement.requirement_id in self.related_requirements or
            self.requirement_id in other_requirement.related_requirements or
            self.ifrs_reference == other_requirement.ifrs_reference
        )


class IFRSRequirementGroup(BaseModel):
    """Represents a group of related IFRS requirements."""
    
    group_id: str = Field(..., description="Unique identifier for the requirement group")
    group_name: str = Field(..., description="Name of the requirement group")
    category: str = Field(..., description="Category this group belongs to")
    
    # Group details
    description: str = Field(..., description="Description of what this group covers")
    requirements: List[IFRSRequirement] = Field(..., description="Requirements in this group")
    
    # Group metadata
    priority: str = Field(..., description="Priority level: low, medium, high, critical")
    complexity: str = Field(..., description="Complexity level: simple, moderate, complex")
    
    # Relationships
    parent_group: Optional[str] = Field(None, description="Parent group ID if this is a subgroup")
    child_groups: List[str] = Field(default_factory=list, description="Child group IDs")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_requirement_count(self) -> int:
        """Get the total number of requirements in this group."""
        return len(self.requirements)
    
    def get_mandatory_requirements(self) -> List[IFRSRequirement]:
        """Get all mandatory requirements in this group."""
        return [req for req in self.requirements if req.mandatory]
    
    def get_requirements_by_type(self, requirement_type: str) -> List[IFRSRequirement]:
        """Get requirements of a specific type."""
        return [req for req in self.requirements if req.requirement_type == requirement_type]


class IFRSStandard(BaseModel):
    """Represents a complete IFRS standard with all its requirements."""
    
    standard_id: str = Field(..., description="IFRS standard identifier (e.g., IFRS 15)")
    standard_name: str = Field(..., description="Full name of the IFRS standard")
    version: str = Field(..., description="Version of the standard")
    effective_date: str = Field(..., description="Effective date of the standard")
    
    # Standard structure
    categories: List[str] = Field(..., description="Main categories in the standard")
    requirement_groups: List[IFRSRequirementGroup] = Field(..., description="Requirement groups")
    all_requirements: List[IFRSRequirement] = Field(..., description="All individual requirements")
    
    # Standard metadata
    description: str = Field(..., description="Description of what the standard covers")
    scope: str = Field(..., description="Scope of the standard")
    key_principles: List[str] = Field(..., description="Key principles of the standard")
    
    # Processing metadata
    extraction_source: str = Field(..., description="Source of the standard extraction")
    last_updated: str = Field(..., description="Last update date")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_requirement_by_id(self, requirement_id: str) -> Optional[IFRSRequirement]:
        """Get a requirement by its ID."""
        for req in self.all_requirements:
            if req.requirement_id == requirement_id:
                return req
        return None
    
    def get_requirements_by_category(self, category: str) -> List[IFRSRequirement]:
        """Get all requirements in a specific category."""
        return [req for req in self.all_requirements if req.category == category]
    
    def get_requirements_by_reference(self, ifrs_reference: str) -> List[IFRSRequirement]:
        """Get all requirements for a specific IFRS reference."""
        return [req for req in self.all_requirements if req.ifrs_reference == ifrs_reference]
    
    def get_mandatory_requirements(self) -> List[IFRSRequirement]:
        """Get all mandatory requirements in the standard."""
        return [req for req in self.all_requirements if req.mandatory]
    
    def get_requirement_count(self) -> int:
        """Get the total number of requirements in the standard."""
        return len(self.all_requirements) 