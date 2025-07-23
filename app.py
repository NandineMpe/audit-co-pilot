#!/usr/bin/env python3
"""
Audit Co-Pilot - Compliance Checklist Management System
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import uvicorn
import os
import json
from enum import Enum

app = FastAPI(
    title="Audit Co-Pilot", 
    version="1.0.0",
    description="Comprehensive compliance checklist and audit management system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"

class ComplianceItem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    requirement: str
    status: ComplianceStatus
    evidence: Optional[str] = None
    notes: Optional[str] = None
    last_updated: datetime
    assigned_to: Optional[str] = None
    due_date: Optional[date] = None
    risk_level: str = "medium"

class ComplianceChecklist(BaseModel):
    id: str
    name: str
    description: str
    framework: str
    version: str
    items: List[ComplianceItem]
    created_date: datetime
    last_audit_date: Optional[datetime] = None
    overall_status: ComplianceStatus

class AuditLog(BaseModel):
    id: str
    checklist_id: str
    action: str
    details: str
    user: str
    timestamp: datetime
    changes: Optional[Dict[str, Any]] = None

# In-memory storage (in production, use a database)
compliance_checklists: Dict[str, ComplianceChecklist] = {}
audit_logs: List[AuditLog] = []

# Sample compliance frameworks
SAMPLE_FRAMEWORKS = {
    "SOC2": {
        "name": "SOC 2 Type II",
        "categories": ["Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"]
    },
    "ISO27001": {
        "name": "ISO 27001 Information Security",
        "categories": ["Information Security Management", "Asset Management", "Access Control", "Incident Management"]
    },
    "GDPR": {
        "name": "General Data Protection Regulation",
        "categories": ["Data Protection", "Privacy Rights", "Data Processing", "Breach Notification"]
    },
    "HIPAA": {
        "name": "Health Insurance Portability and Accountability Act",
        "categories": ["Privacy Rule", "Security Rule", "Breach Notification", "Enforcement"]
    }
}

# Sample compliance items
SAMPLE_ITEMS = {
    "SOC2": [
        {
            "id": "SOC2-001",
            "title": "Access Control Policy",
            "description": "Implement and maintain access control policies and procedures",
            "category": "Security",
            "requirement": "CC6.1",
            "status": ComplianceStatus.COMPLIANT,
            "evidence": "Access control policy document v2.1",
            "risk_level": "high"
        },
        {
            "id": "SOC2-002", 
            "title": "Data Encryption",
            "description": "Encrypt sensitive data at rest and in transit",
            "category": "Security",
            "requirement": "CC6.8",
            "status": ComplianceStatus.PARTIAL,
            "evidence": "AES-256 encryption implemented for data at rest",
            "notes": "Need to implement TLS 1.3 for data in transit",
            "risk_level": "high"
        },
        {
            "id": "SOC2-003",
            "title": "Incident Response Plan",
            "description": "Maintain incident response procedures and team",
            "category": "Security", 
            "requirement": "CC7.4",
            "status": ComplianceStatus.NON_COMPLIANT,
            "evidence": None,
            "notes": "Incident response plan needs to be developed",
            "risk_level": "critical"
        }
    ]
}

@app.get("/")
async def root():
    return {
        "message": "Audit Co-Pilot Compliance System is running!", 
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "checklists": "/api/checklists",
            "frameworks": "/api/frameworks", 
            "audit-logs": "/api/audit-logs",
            "reports": "/api/reports"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "audit-co-pilot"}

# Compliance Framework Endpoints
@app.get("/api/frameworks")
async def get_frameworks():
    """Get available compliance frameworks"""
    return {"frameworks": SAMPLE_FRAMEWORKS}

@app.get("/api/frameworks/{framework_id}")
async def get_framework(framework_id: str):
    """Get specific framework details"""
    if framework_id not in SAMPLE_FRAMEWORKS:
        raise HTTPException(status_code=404, detail="Framework not found")
    return {"framework": SAMPLE_FRAMEWORKS[framework_id]}

# Compliance Checklist Endpoints
@app.get("/api/checklists")
async def get_checklists():
    """Get all compliance checklists"""
    return {"checklists": list(compliance_checklists.values())}

@app.get("/api/checklists/{checklist_id}")
async def get_checklist(checklist_id: str):
    """Get specific compliance checklist"""
    if checklist_id not in compliance_checklists:
        raise HTTPException(status_code=404, detail="Checklist not found")
    return {"checklist": compliance_checklists[checklist_id]}

@app.post("/api/checklists")
async def create_checklist(checklist: ComplianceChecklist):
    """Create a new compliance checklist"""
    compliance_checklists[checklist.id] = checklist
    
    # Log the creation
    audit_log = AuditLog(
        id=f"log_{len(audit_logs) + 1}",
        checklist_id=checklist.id,
        action="CREATE",
        details=f"Created checklist: {checklist.name}",
        user="system",
        timestamp=datetime.now()
    )
    audit_logs.append(audit_log)
    
    return {"message": "Checklist created successfully", "checklist": checklist}

@app.put("/api/checklists/{checklist_id}/items/{item_id}")
async def update_compliance_item(checklist_id: str, item_id: str, item: ComplianceItem):
    """Update a compliance item"""
    if checklist_id not in compliance_checklists:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    checklist = compliance_checklists[checklist_id]
    item_index = None
    
    for i, existing_item in enumerate(checklist.items):
        if existing_item.id == item_id:
            item_index = i
            break
    
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Store old status for audit log
    old_status = checklist.items[item_index].status
    
    # Update the item
    checklist.items[item_index] = item
    checklist.last_updated = datetime.now()
    
    # Update overall status
    update_overall_status(checklist)
    
    # Log the change
    audit_log = AuditLog(
        id=f"log_{len(audit_logs) + 1}",
        checklist_id=checklist_id,
        action="UPDATE_ITEM",
        details=f"Updated item {item_id}: {old_status} -> {item.status}",
        user="system",
        timestamp=datetime.now(),
        changes={"item_id": item_id, "old_status": old_status, "new_status": item.status}
    )
    audit_logs.append(audit_log)
    
    return {"message": "Item updated successfully", "item": item}

# Audit Log Endpoints
@app.get("/api/audit-logs")
async def get_audit_logs(checklist_id: Optional[str] = None, limit: int = 50):
    """Get audit logs with optional filtering"""
    logs = audit_logs
    if checklist_id:
        logs = [log for log in logs if log.checklist_id == checklist_id]
    
    return {"audit_logs": logs[-limit:]}

# Reporting Endpoints
@app.get("/api/reports/compliance-summary")
async def get_compliance_summary():
    """Get overall compliance summary"""
    total_checklists = len(compliance_checklists)
    total_items = sum(len(checklist.items) for checklist in compliance_checklists.values())
    
    status_counts = {"compliant": 0, "non_compliant": 0, "partial": 0, "not_applicable": 0, "pending": 0}
    
    for checklist in compliance_checklists.values():
        for item in checklist.items:
            status_counts[item.status.value] += 1
    
    compliance_rate = (status_counts["compliant"] / total_items * 100) if total_items > 0 else 0
    
    return {
        "summary": {
            "total_checklists": total_checklists,
            "total_items": total_items,
            "compliance_rate": round(compliance_rate, 2),
            "status_breakdown": status_counts
        }
    }

@app.get("/api/reports/risk-assessment")
async def get_risk_assessment():
    """Get risk assessment report"""
    risk_items = []
    
    for checklist in compliance_checklists.values():
        for item in checklist.items:
            if item.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL]:
                risk_items.append({
                    "checklist_id": checklist.id,
                    "checklist_name": checklist.name,
                    "item_id": item.id,
                    "item_title": item.title,
                    "status": item.status,
                    "risk_level": item.risk_level,
                    "category": item.category
                })
    
    # Sort by risk level
    risk_level_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    risk_items.sort(key=lambda x: risk_level_order.get(x["risk_level"], 0), reverse=True)
    
    return {"risk_assessment": risk_items}

# Sample Data Endpoints
@app.post("/api/sample-data/soc2")
async def create_sample_soc2_checklist():
    """Create a sample SOC2 compliance checklist"""
    items = []
    for item_data in SAMPLE_ITEMS["SOC2"]:
        item = ComplianceItem(
            **item_data,
            last_updated=datetime.now(),
            assigned_to="Security Team"
        )
        items.append(item)
    
    checklist = ComplianceChecklist(
        id="SOC2-2024",
        name="SOC 2 Type II Compliance Checklist",
        description="Comprehensive SOC 2 Type II compliance assessment",
        framework="SOC2",
        version="2024",
        items=items,
        created_date=datetime.now(),
        overall_status=ComplianceStatus.PARTIAL
    )
    
    compliance_checklists[checklist.id] = checklist
    
    # Log creation
    audit_log = AuditLog(
        id=f"log_{len(audit_logs) + 1}",
        checklist_id=checklist.id,
        action="CREATE_SAMPLE",
        details="Created sample SOC2 checklist",
        user="system",
        timestamp=datetime.now()
    )
    audit_logs.append(audit_log)
    
    return {"message": "Sample SOC2 checklist created", "checklist": checklist}

def update_overall_status(checklist: ComplianceChecklist):
    """Update the overall compliance status of a checklist"""
    if not checklist.items:
        checklist.overall_status = ComplianceStatus.PENDING
        return
    
    status_counts = {"compliant": 0, "non_compliant": 0, "partial": 0, "not_applicable": 0, "pending": 0}
    
    for item in checklist.items:
        status_counts[item.status.value] += 1
    
    total_items = len(checklist.items)
    
    if status_counts["non_compliant"] > 0:
        checklist.overall_status = ComplianceStatus.NON_COMPLIANT
    elif status_counts["partial"] > 0:
        checklist.overall_status = ComplianceStatus.PARTIAL
    elif status_counts["compliant"] == total_items:
        checklist.overall_status = ComplianceStatus.COMPLIANT
    else:
        checklist.overall_status = ComplianceStatus.PENDING

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)