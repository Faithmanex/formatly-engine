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
from datetime import datetime, timedelta
import asyncio
import jwt
import httpx
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_JWT_SECRET, SUPABASE_ANON_KEY]):
    raise ValueError("Missing required Supabase environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

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

class CreateUploadResponse(BaseModel):
    success: bool
    job_id: str
    upload_url: str
    upload_token: str
    file_path: str
    message: str
    upload_headers: Dict[str, str]

class WebhookUploadComplete(BaseModel):
    job_id: str
    file_path: str
    success: bool

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
    """Verify JWT token with Supabase"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Decode JWT token
        token = credentials.credentials
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=["HS256"],
            audience="authenticated"
        )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        
        # Get user profile from database
        profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
        
        if not profile_response.data:
            raise HTTPException(status_code=401, detail="User profile not found")
        
        profile = profile_response.data[0]
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "profile": profile
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(status_code=401, detail="Token verification failed")

async def generate_signed_upload_url(filename: str, user_id: str) -> Dict[str, str]:
    """Generate signed upload URL for document upload to Supabase Storage and include required headers for frontend"""
    try:
        # Create unique file path
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'docx'
        unique_filename = f"{user_id}/{uuid.uuid4()}.{file_extension}"
        logger.error(f"Unique filename: {unique_filename}")
        # Generate signed upload URL
        response = supabase.storage.from_("documents").create_signed_upload_url(unique_filename)
        
        if not response or not response.get("signedURL"):
            raise Exception("Failed to generate signed upload URL")
        
        upload_headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/octet-stream"
        }
        
        return {
            "upload_url": response["signedURL"],
            "file_path": unique_filename,
            "upload_token": response.get("token", ""),
            "upload_headers": upload_headers
        }
        
    except Exception as e:
        logger.error(f"Error generating signed upload URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")

# Keep the old function for backward compatibility if needed
async def generate_presigned_upload_url(filename: str, user_id: str) -> Dict[str, str]:
    """Legacy function - redirects to corrected version"""
    return await generate_signed_upload_url(filename, user_id)

async def create_document_record(user_id: str, filename: str, file_path: str, job_id: str, style: str, language_variant: str, options: Dict[str, Any]) -> str:
    """Create document record in Supabase database"""
    try:
        document_data = {
            "id": job_id,
            "user_id": user_id,
            "filename": filename,
            "original_filename": filename,
            "status": "draft",
            "style_applied": style,
            "language_variant": language_variant,
            "storage_location": file_path,
            "formatting_options": options,
            "report_only": options.get("reportOnly", False),
            "file_type": filename.split('.')[-1].lower() if '.' in filename else "docx",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table("documents").insert(document_data).execute()
        
        if not response.data:
            raise Exception("Failed to create document record")
        
        return job_id
        
    except Exception as e:
        logger.error(f"Error creating document record: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {str(e)}")

async def update_document_status(job_id: str, status: str, progress: int = None, processing_log: Dict[str, Any] = None):
    """Update document status in database"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            if not processing_log:
                processing_log = {}
            processing_log["progress"] = progress
            update_data["processing_log"] = processing_log
        
        if status == "formatted":
            update_data["processed_at"] = datetime.utcnow().isoformat()
        
        response = supabase.table("documents").update(update_data).eq("id", job_id).execute()
        
        if not response.data:
            logger.warning(f"No document found with ID: {job_id}")
        
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")

async def simulate_processing(job_id: str):
    """Simulate document processing with database updates"""
    try:
        await asyncio.sleep(1)  # Initial delay
        
        # Update to processing
        await update_document_status(job_id, "processing", 25)
        
        await asyncio.sleep(2)  # Processing time
        await update_document_status(job_id, "processing", 75)
        
        await asyncio.sleep(1)  # Final processing
        
        # Mock processing results
        processing_log = {
            "progress": 100,
            "word_count": 1250,
            "headings_count": 8,
            "references_count": 15,
            "processing_time": 3.5
        }
        
        await update_document_status(job_id, "formatted", 100, processing_log)
        
    except Exception as e:
        logger.error(f"Error in processing simulation: {str(e)}")
        await update_document_status(job_id, "failed", processing_log={"error": str(e)})

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

@app.post("/api/documents/create-upload")
async def create_upload_url(
    filename: str = Form(...),
    style: str = Form(...),
    englishVariant: str = Form(...),
    reportOnly: bool = Form(False),
    includeComments: bool = Form(True),
    preserveFormatting: bool = Form(True),
    user: dict = Depends(verify_token)
) -> CreateUploadResponse:
    """NEW ENDPOINT: Generate signed upload URL and create document record - PROPER FLOW"""
    try:
        job_id = str(uuid.uuid4())
        
        # STEP 1: Generate signed upload URL first
        upload_info = await generate_signed_upload_url(filename, user["user_id"])
        
        # STEP 2: Create document record with DRAFT status
        options = {
            "reportOnly": reportOnly,
            "includeComments": includeComments,
            "preserveFormatting": preserveFormatting
        }
        
        await create_document_record(
            user["user_id"], 
            filename, 
            upload_info["file_path"], 
            job_id, 
            style, 
            englishVariant, 
            options
        )
        
        # STEP 3: Return job ID and upload URL to frontend
        return CreateUploadResponse(
            success=True,
            job_id=job_id,
            upload_url=upload_info["upload_url"],
            upload_token=upload_info["upload_token"],
            file_path=upload_info["file_path"],
            upload_headers=upload_info["upload_headers"],
            message=f"Upload URL created. Document status: DRAFT. Please upload your document to begin {style} formatting."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create upload URL error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create upload URL: {str(e)}")

@app.post("/api/documents/upload-complete")
async def upload_complete_webhook(
    webhook_data: WebhookUploadComplete,
    user: dict = Depends(verify_token)
):
    """NEW ENDPOINT: Webhook called after frontend successfully uploads file"""
    try:
        job_id = webhook_data.job_id
        
        # Verify the job belongs to the user
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        document = response.data[0]
        
        if webhook_data.success:
            # Update status to processing and start processing
            await update_document_status(job_id, "processing", 0)
            
            # Start background processing
            asyncio.create_task(simulate_processing(job_id))
            
            return {
                "success": True,
                "message": "Upload confirmed. Document processing started. Status: PROCESSING",
                "job_id": job_id
            }
        else:
            # Upload failed
            await update_document_status(job_id, "failed", processing_log={"error": "File upload failed"})
            
            return {
                "success": False,
                "message": "Upload failed. Status: FAILED. Please try again.",
                "job_id": job_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload complete webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.post("/api/documents/upload")
async def upload_document(
    filename: str = Form(...),
    style: str = Form(...),
    englishVariant: str = Form(...),
    reportOnly: bool = Form(False),
    includeComments: bool = Form(True),
    preserveFormatting: bool = Form(True),
    user: dict = Depends(verify_token)
):
    """EXISTING ENDPOINT: Generate presigned upload URL and create document record - KEPT FOR COMPATIBILITY"""
    try:
        job_id = str(uuid.uuid4())
        
        # FIXED: Use corrected upload URL generation
        upload_info = await generate_signed_upload_url(filename, user["user_id"])
        
        # Prepare formatting options
        options = {
            "reportOnly": reportOnly,
            "includeComments": includeComments,
            "preserveFormatting": preserveFormatting
        }
        
        # Create document record in database
        await create_document_record(
            user["user_id"], 
            filename, 
            upload_info["file_path"], 
            job_id, 
            style, 
            englishVariant, 
            options
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "upload_url": upload_info["upload_url"],
            "upload_token": upload_info.get("upload_token", ""),
            "file_path": upload_info["file_path"],
            "upload_headers": upload_info["upload_headers"],
            "message": f"Document queued for {style} formatting. Status: DRAFT. Upload your file to begin processing."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/documents/process")
async def process_document(
    request: ProcessDocumentRequest,
    user: dict = Depends(verify_token)
):
    """Process a document with formatting (legacy endpoint for backward compatibility)"""
    try:
        job_id = str(uuid.uuid4())
        
        # For backward compatibility, create a document record with base64 content
        options = {
            "reportOnly": request.reportOnly,
            "includeComments": request.includeComments,
            "preserveFormatting": request.preserveFormatting,
            "content": request.content  # Store base64 content temporarily
        }
        
        await create_document_record(
            user["user_id"],
            request.filename,
            f"legacy/{job_id}",  # Legacy storage path
            job_id,
            request.style,
            request.englishVariant,
            options
        )
        
        # Start background processing immediately for legacy endpoint
        asyncio.create_task(simulate_processing(job_id))
        
        return ProcessDocumentResponse(
            success=True,
            job_id=job_id,
            status="draft",
            message=f"Document queued for {request.style} formatting"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/documents/status/{job_id}")
async def get_document_status(job_id: str, user: dict = Depends(verify_token)):
    """Get the status of a document processing job from database"""
    try:
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        document = response.data[0]
        processing_log = document.get("processing_log") or {}
        
        return DocumentStatusResponse(
            success=True,
            job_id=job_id,
            status=document["status"],
            progress=processing_log.get("progress", 0),
            result_url=f"/api/documents/download/{job_id}" if document["status"] == "formatted" else None,
            error=processing_log.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/api/documents/download/{job_id}")
async def download_document(job_id: str, user: dict = Depends(verify_token)):
    """Download a formatted document"""
    try:
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        document = response.data[0]
        
        if document["status"] != "formatted":
            raise HTTPException(status_code=400, detail="Document not ready for download")
        
        # For now, return mock formatted content since actual formatting is not implemented
        processing_log = document.get("processing_log") or {}
        
        # Mock formatted content
        mock_content = f"FORMATTED DOCUMENT: {document['filename']}\nStyle: {document['style_applied']}\nVariant: {document['language_variant']}"
        formatted_content = base64.b64encode(mock_content.encode()).decode()
        
        metadata = {
            "word_count": processing_log.get("word_count", 1250),
            "headings_count": processing_log.get("headings_count", 8),
            "references_count": processing_log.get("references_count", 15),
            "style_applied": document["style_applied"],
            "processing_time": processing_log.get("processing_time", 3.5)
        }
        
        return FormattedDocumentResponse(
            success=True,
            filename=f"formatted_{document['filename']}",
            content=formatted_content,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/api/formatting/styles")
async def get_formatting_styles():
    """Get available formatting styles from database"""
    try:
        response = supabase.table("active_formatting_styles").select("*").order("sort_order").execute()
        
        if response.data:
            return [
                {
                    "id": style["code"],
                    "name": style["name"],
                    "description": style["description"]
                }
                for style in response.data
            ]
        else:
            # Fallback to mock data if database is empty
            return FORMATTING_STYLES
            
    except Exception as e:
        logger.error(f"Error fetching formatting styles: {str(e)}")
        return FORMATTING_STYLES

@app.get("/api/formatting/variants")
async def get_english_variants():
    """Get available English variants from database"""
    try:
        response = supabase.table("active_english_variants").select("*").order("sort_order").execute()
        
        if response.data:
            return [
                {
                    "id": variant["code"],
                    "name": variant["name"],
                    "description": variant["description"]
                }
                for variant in response.data
            ]
        else:
            # Fallback to mock data if database is empty
            return ENGLISH_VARIANTS
            
    except Exception as e:
        logger.error(f"Error fetching English variants: {str(e)}")
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
