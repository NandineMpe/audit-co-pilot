"""
Checklist Parser for IFRS Compliance Requirements.

This module provides functionality to parse IFRS compliance checklists
from Excel files and extract structured requirement data.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from ..models.ifrs_requirement import IFRSRequirement, IFRSRequirementGroup, IFRSStandard

logger = logging.getLogger(__name__)


class ChecklistParser:
    """
    Parser for IFRS compliance checklists in Excel format.
    
    This class handles the extraction and structuring of IFRS compliance
    requirements from Excel-based checklists.
    """
    
    def __init__(self, config=None):
        """Initialize the checklist parser."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Column mapping for the actual IFRS checklist format
        self.column_mapping = {
            'reference': 'Index',
            'requirement_text': 'Unnamed: 10',
            'category': 'sheet_name',  # We'll derive this from sheet name
            'standard': 'sheet_name'   # We'll derive this from sheet name
        }
    
    def parse_checklist_file(self, file_path: str) -> IFRSStandard:
        """
        Parse an IFRS compliance checklist file and extract requirements.
        
        Args:
            file_path: Path to the Excel checklist file
            
        Returns:
            IFRSStandard object containing all extracted requirements
        """
        try:
            self.logger.info(f"Parsing checklist file: {file_path}")
            
            # Read all sheets from the Excel file
            all_sheets = pd.read_excel(file_path, sheet_name=None)
            
            # Filter for IFRS sheets (those starting with IFRS and <= 6 chars)
            ifrs_sheets = {name: df for name, df in all_sheets.items() 
                          if name.startswith('IFRS') and len(name) <= 6}
            
            self.logger.info(f"Found {len(ifrs_sheets)} IFRS standard sheets")
            
            all_requirements = []
            
            # Process each IFRS sheet
            for sheet_name, df in ifrs_sheets.items():
                self.logger.info(f"Processing sheet: {sheet_name}")
                sheet_requirements = self._parse_sheet(df, sheet_name)
                all_requirements.extend(sheet_requirements)
            
            if not all_requirements:
                raise ValueError("No requirements found to create standard")
            
            # Create IFRS standard object
            standard = self._create_ifrs_standard(all_requirements, file_path)
            
            self.logger.info(f"Successfully parsed {len(all_requirements)} requirements")
            return standard
            
        except Exception as e:
            self.logger.error(f"Error parsing checklist file: {e}")
            raise
    
    def _parse_sheet(self, df: pd.DataFrame, sheet_name: str) -> List[IFRSRequirement]:
        """
        Parse a single sheet and extract requirements.
        
        Args:
            df: DataFrame containing the sheet data
            sheet_name: Name of the sheet (e.g., 'IFRS1A')
            
        Returns:
            List of IFRSRequirement objects
        """
        requirements = []
        
        for idx, row in df.iterrows():
            try:
                # Look for rows with IFRS references in the Index column
                index_val = str(row.get('Index', ''))
                if 'IFRS' in index_val and ':' in index_val:
                    # Get the requirement text from the Unnamed: 10 column
                    req_text = str(row.get('Unnamed: 10', ''))
                    if req_text and req_text.strip() and req_text != 'nan':
                        requirement = self._create_requirement(
                            index_val.strip(),
                            req_text.strip(),
                            sheet_name,
                            idx
                        )
                        if requirement:
                            requirements.append(requirement)
            except Exception as e:
                self.logger.warning(f"Error parsing row {idx} in {sheet_name}: {e}")
                continue
        
        return requirements
    
    def _create_requirement(self, reference: str, requirement_text: str, 
                          sheet_name: str, row_idx: int) -> Optional[IFRSRequirement]:
        """
        Create an IFRSRequirement object from parsed data.
        
        Args:
            reference: IFRS reference (e.g., 'IFRS 1:2')
            requirement_text: The requirement text
            sheet_name: Name of the sheet
            row_idx: Row index in the sheet
            
        Returns:
            IFRSRequirement object or None if invalid
        """
        try:
            # Parse the reference to extract standard and section
            standard_match = re.match(r'(IFRS\s+\d+)', reference)
            if not standard_match:
                return None
            
            standard_name = standard_match.group(1)
            
            # Determine category based on sheet name
            category = self._determine_category(sheet_name)
            
            # Create requirement ID
            requirement_id = f"{reference.replace(' ', '_').replace(':', '_')}_{sheet_name}"
            
            requirement = IFRSRequirement(
                requirement_id=requirement_id,
                requirement_text=requirement_text,
                ifrs_reference=reference,
                category=category,
                subcategory=sheet_name,
                section=reference.split(':')[1] if ':' in reference else None,
                requirement_type=self._determine_requirement_type(requirement_text),
                mandatory=True,  # Assume mandatory unless specified otherwise
                materiality_threshold=None,
                guidance_notes="",
                examples=[],
                related_requirements=[],
                checklist_source=sheet_name,
                version="2024",
                effective_date="2024-01-01",
                extraction_confidence=0.9,
                processing_notes=[f"Extracted from row {row_idx}"]
            )
            
            return requirement
            
        except Exception as e:
            self.logger.warning(f"Error creating requirement for {reference}: {e}")
            return None
    
    def _determine_category(self, sheet_name: str) -> str:
        """Determine the category based on sheet name."""
        if sheet_name.endswith('A'):
            return "Accounting"
        elif sheet_name.endswith('P'):
            return "Presentation"
        else:
            return "General"
    
    def _determine_requirement_type(self, requirement_text: str) -> str:
        """Determine the requirement type based on the text."""
        text_lower = requirement_text.lower()
        
        if any(word in text_lower for word in ['disclose', 'disclosure', 'present', 'presentation']):
            return "disclosure"
        elif any(word in text_lower for word in ['measure', 'measurement', 'calculate', 'value']):
            return "measurement"
        elif any(word in text_lower for word in ['apply', 'implement', 'follow']):
            return "presentation"
        else:
            return "general"
    
    def _create_ifrs_standard(self, requirements: List[IFRSRequirement], 
                            file_path: str) -> IFRSStandard:
        """
        Create an IFRSStandard object from extracted requirements.
        
        Args:
            requirements: List of IFRSRequirement objects
            file_path: Source file path
            
        Returns:
            IFRSStandard object
        """
        # Group requirements by category
        categories = {}
        for req in requirements:
            cat = req.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(req)
        
        # Create requirement groups
        requirement_groups = []
        for category, reqs in categories.items():
            group = IFRSRequirementGroup(
                group_id=f"group_{category.lower()}",
                group_name=f"{category} Requirements",
                category=category,
                description=f"Requirements related to {category.lower()}",
                requirements=reqs,
                priority="high",
                complexity="moderate",
                parent_group=None,
                child_groups=[]
            )
            requirement_groups.append(group)
        
        # Create the standard
        standard = IFRSStandard(
            standard_id="IFRS_Comprehensive",
            standard_name="Comprehensive IFRS Standards",
            version="2024",
            effective_date="2024-01-01",
            categories=list(categories.keys()),
            requirement_groups=requirement_groups,
            all_requirements=requirements,
            description="Comprehensive IFRS compliance requirements extracted from checklist",
            scope="All IFRS standards covered in the checklist",
            key_principles=["Compliance", "Transparency", "Accuracy"],
            extraction_source=file_path,
            last_updated="2024-01-01"
        )
        
        return standard
    
    def validate_checklist(self, file_path: str) -> List[str]:
        """
        Validate the checklist file and return validation errors.
        
        Args:
            file_path: Path to the checklist file
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Check if file exists
            if isinstance(file_path, str):
                file_path_obj = Path(file_path)
            else:
                file_path_obj = file_path
                
            if not file_path_obj.exists():
                errors.append(f"Checklist file not found: {file_path}")
                return errors
            
            # Try to read the file
            all_sheets = pd.read_excel(str(file_path_obj), sheet_name=None)
            
            # Check if we have any IFRS sheets
            ifrs_sheets = {name: df for name, df in all_sheets.items() 
                          if name.startswith('IFRS') and len(name) <= 6}
            
            if not ifrs_sheets:
                errors.append("No IFRS standard sheets found in the checklist")
            
            # Check each IFRS sheet for requirements
            total_requirements = 0
            for sheet_name, df in ifrs_sheets.items():
                sheet_requirements = 0
                for idx, row in df.iterrows():
                    index_val = str(row.get('Index', ''))
                    if 'IFRS' in index_val and ':' in index_val:
                        req_text = str(row.get('Unnamed: 10', ''))
                        if req_text and req_text.strip() and req_text != 'nan':
                            sheet_requirements += 1
                
                if sheet_requirements == 0:
                    errors.append(f"No requirements found in sheet: {sheet_name}")
                else:
                    total_requirements += sheet_requirements
            
            if total_requirements == 0:
                errors.append("No requirements found in any IFRS sheet")
            
        except Exception as e:
            errors.append(f"Error validating checklist: {e}")
        
        return errors

    def validate_requirements(self, requirements: List[IFRSRequirement]) -> List[str]:
        """
        Validate extracted requirements and return validation errors.
        
        Args:
            requirements: List of requirements to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        for req in requirements:
            if not req.requirement_id:
                errors.append(f"Missing requirement ID for {req.ifrs_reference}")
            if not req.requirement_text:
                errors.append(f"Missing requirement text for {req.ifrs_reference}")
            if not req.ifrs_reference:
                errors.append(f"Missing IFRS reference for {req.requirement_id}")
        
        return errors
    
    def export_requirements(self, requirements: List[IFRSRequirement], 
                          output_path: str, format: str = "json") -> None:
        """
        Export requirements to various formats.
        
        Args:
            requirements: List of requirements to export
            output_path: Output file path
            format: Export format ('json', 'csv', 'excel')
        """
        try:
            if format == "json":
                import json
                data = [req.model_dump() for req in requirements]
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif format == "csv":
                data = []
                for req in requirements:
                    data.append({
                        'requirement_id': req.requirement_id,
                        'ifrs_reference': req.ifrs_reference,
                        'requirement_text': req.requirement_text,
                        'category': req.category,
                        'requirement_type': req.requirement_type
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            
            elif format == "excel":
                data = []
                for req in requirements:
                    data.append({
                        'requirement_id': req.requirement_id,
                        'ifrs_reference': req.ifrs_reference,
                        'requirement_text': req.requirement_text,
                        'category': req.category,
                        'requirement_type': req.requirement_type
                    })
                
                df = pd.DataFrame(data)
                df.to_excel(output_path, index=False)
            
            self.logger.info(f"Exported {len(requirements)} requirements to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting requirements: {e}")
            raise 