from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import time
import base64
import json
import os
from datetime import datetime
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="Formatly API",
    description="Document formatting service API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# In-memory storage for demo (use Redis/database in production)
jobs_storage: Dict[str, Dict[str, Any]] = {}
files_storage: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class ProcessDocumentRequest(BaseModel):
    filename: str
    content: str  # base64 encoded
    style: str
    englishVariant: str
    reportOnly: bool = False
    includeComments: bool = True
    preserveFormatting: bool = True
    options: Optional[Dict[str, Any]] = None

class ProcessDocumentResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str

class DocumentStatusResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    progress: int
    result_url: Optional[str] = None
    error: Optional[str] = None

class FormattedDocumentResponse(BaseModel):
    success: bool
    filename: str
    content: str  # base64 encoded
    metadata: Dict[str, Any]

class FormattingStyle(BaseModel):
    id: str
    name: str
    description: str

class EnglishVariant(BaseModel):
    id: str
    name: str
    description: str

# Mock data
FORMATTING_STYLES = [
    {"id": "apa", "name": "APA Style", "description": "American Psychological Association"},
    {"id": "mla", "name": "MLA Style", "description": "Modern Language Association"},
    {"id": "chicago", "name": "Chicago Style", "description": "Chicago Manual of Style"},
    {"id": "ieee", "name": "IEEE Style", "description": "Institute of Electrical and Electronics Engineers"},
]

ENGLISH_VARIANTS = [
    {"id": "us", "name": "US English", "description": "American English"},
    {"id": "uk", "name": "UK English", "description": "British English"},
    {"id": "ca", "name": "Canadian English", "description": "Canadian English"},
    {"id": "au", "name": "Australian English", "description": "Australian English"},
]

# Helper functions
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Mock JWT verification - replace with actual Supabase JWT verification"""
    if not credentials:
        return None
    # In production, verify the JWT token with Supabase
    # For now, just return a mock user
    return {"user_id": "mock_user", "email": "test@example.com"}

async def simulate_processing(job_id: str):
    """Simulate document processing"""
    await asyncio.sleep(1)  # Initial delay
    
    # Update to processing
    jobs_storage[job_id]["status"] = "processing"
    jobs_storage[job_id]["progress"] = 25
    
    await asyncio.sleep(2)  # Processing time
    jobs_storage[job_id]["progress"] = 75
    
    await asyncio.sleep(1)  # Final processing
    jobs_storage[job_id]["status"] = "formatted"
    jobs_storage[job_id]["progress"] = 100
    jobs_storage[job_id]["result_url"] = f"/api/documents/download/{job_id}"
    
    # Create mock formatted document
    original_content = jobs_storage[job_id]["original_content"]
    formatted_content = f"FORMATTED: {original_content}"  # Mock formatting
    
    jobs_storage[job_id]["formatted_content"] = base64.b64encode(
        formatted_content.encode()
    ).decode()

# API Routes

@app.get("/")
async def root():
    return {
        "message": "Formatly API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(verify_token)
):
    """Upload a document for processing"""
    try:
        # Read file content
        content = await file.read()
        file_id = str(uuid.uuid4())
        
        # Store file info
        files_storage[file_id] = {
            "filename": file.filename,
            "content": base64.b64encode(content).decode(),
            "content_type": file.content_type,
            "size": len(content),
            "uploaded_at": datetime.utcnow().isoformat(),
            "user_id": user["user_id"] if user else "anonymous"
        }
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/documents/process")
async def process_document(
    request: ProcessDocumentRequest,
    user: Optional[dict] = Depends(verify_token)
):
    """Process a document with formatting"""
    try:
        job_id = str(uuid.uuid4())
        
        # Store job info
        jobs_storage[job_id] = {
            "job_id": job_id,
            "filename": request.filename,
            "original_content": request.content,
            "style": request.style,
            "englishVariant": request.englishVariant,
            "reportOnly": request.reportOnly,
            "includeComments": request.includeComments,
            "preserveFormatting": request.preserveFormatting,
            "options": request.options or {},
            "status": "queued",
            "progress": 0,
            "created_at": datetime.utcnow().isoformat(),
            "user_id": user["user_id"] if user else "anonymous"
        }
        
        # Start background processing
        asyncio.create_task(simulate_processing(job_id))
        
        return ProcessDocumentResponse(
            success=True,
            job_id=job_id,
            status="queued",
            message=f"Document queued for {request.style} formatting"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/documents/status/{job_id}")
async def get_document_status(job_id: str):
    """Get the status of a document processing job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    return DocumentStatusResponse(
        success=True,
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result_url=job.get("result_url"),
        error=job.get("error")
    )

@app.get("/api/documents/download/{job_id}")
async def download_document(job_id: str):
    """Download a formatted document"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    if job["status"] != "formatted":
        raise HTTPException(status_code=400, detail="Document not ready for download")
    
    # Mock metadata
    metadata = {
        "word_count": 1250,
        "headings_count": 8,
        "references_count": 15,
        "style_applied": job["style"],
        "processing_time": 3.5
    }
    
    return FormattedDocumentResponse(
        success=True,
        filename=f"formatted_{job['filename']}",
        content=job["formatted_content"],
        metadata=metadata
    )

@app.get("/api/formatting/styles")
async def get_formatting_styles():
    """Get available formatting styles"""
    return FORMATTING_STYLES

@app.get("/api/formatting/variants")
async def get_english_variants():
    """Get available English variants"""
    return ENGLISH_VARIANTS

# Additional utility endpoints for testing

@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (for testing)"""
    return {
        "jobs": list(jobs_storage.values()),
        "total": len(jobs_storage)
    }

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job (for testing)"""
    if job_id in jobs_storage:
        del jobs_storage[job_id]
        return {"success": True, "message": "Job deleted"}
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/files")
async def list_files():
    """List all uploaded files (for testing)"""
    return {
        "files": list(files_storage.values()),
        "total": len(files_storage)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
