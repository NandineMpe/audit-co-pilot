#!/usr/bin/env python3
"""
Audit Co-Pilot - Simple FastAPI App
"""

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Audit Co-Pilot", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Audit Co-Pilot is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "audit-co-pilot"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)