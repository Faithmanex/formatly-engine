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
    """Generate signed upload URL for document upload to Supabase Storage"""
    try:
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename: must be between 1-255 characters")
        
        if not user_id or len(user_id) > 255:
            raise ValueError("Invalid user_id")
        
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'docx'
        
        allowed_extensions = {'docx', 'doc', 'pdf', 'txt', 'rtf', 'odt', 'xlsx', 'xls', 'csv'}
        if file_extension not in allowed_extensions:
            raise ValueError(f"File type .{file_extension} not allowed. Allowed types: {', '.join(allowed_extensions)}")
        
        timestamp = int(time.time())
        unique_filename = f"documents/{user_id}/{timestamp}_{uuid.uuid4()}.{file_extension}"
        
        logger.info(f"[v0] Generating signed upload URL for: {unique_filename}")
        
        storage_bucket = "documents"
        
        try:
            storage = supabase.storage.from_(storage_bucket)
            response = storage.create_signed_upload_url(unique_filename)
            
            logger.info(f"[v0] Supabase response type: {type(response)}, content: {str(response)[:200]}")
            
            # Handle response
            if isinstance(response, dict):
                signed_url = response.get("signedURL") or response.get("signed_url") or response.get("url")
                token = response.get("token", "")
            else:
                signed_url = str(response) if response else None
                token = ""
            
            if not signed_url:
                raise ValueError(f"No signed URL in Supabase response: {response}")
            
            logger.info(f"[v0] Successfully generated signed URL for user {user_id}")
            
            upload_headers = {
                "x-upsert": "true"
            }
            
            return {
                "upload_url": signed_url,
                "file_path": unique_filename,
                "upload_token": token,
                "upload_headers": upload_headers
            }
            
        except Exception as storage_error:
            logger.error(f"[v0] Supabase storage error: {str(storage_error)}, type: {type(storage_error).__name__}")
            raise ValueError(f"Failed to generate upload URL: {str(storage_error)}")
        
    except ValueError as ve:
        logger.error(f"[v0] Validation error in signed URL generation: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[v0] Unexpected error generating signed URL: {str(e)}, type: {type(e).__name__}")
        logger.error(f"[v0] Supabase URL set: {bool(SUPABASE_URL)}, bucket: documents")
        raise HTTPException(status_code=500, detail="Failed to generate upload URL. Please contact support.")

async def generate_presigned_upload_url(filename: str, user_id: str) -> Dict[str, str]:
    """Legacy function - redirects to corrected version"""
    return await generate_signed_upload_url(filename, user_id)

async def create_document_record(user_id: str, filename: str, file_path: str, job_id: str, style: str, language_variant: str, options: Dict[str, Any], file_size: Optional[int] = None) -> str:
    """Create document record in Supabase database with retry logic"""
    try:
        if not all([user_id, filename, job_id, style, language_variant]):
            raise ValueError("Missing required fields for document record")
        
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
            "file_type": file_extension if (file_extension := filename.split('.')[-1].lower() if '.' in filename else "docx") else "docx",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if file_size is not None:
            document_data["file_size"] = file_size
        
        logger.info(f"[v0] Creating document record: job_id={job_id}, user_id={user_id}, file={filename}")
        
        response = supabase.table("documents").insert(document_data).execute()
        
        if not response.data:
            raise ValueError("Database insert returned no data")
        
        logger.info(f"[v0] Document record created successfully: {job_id}")
        return job_id
        
    except ValueError as ve:
        logger.error(f"[v0] Validation error creating document: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[v0] Error creating document record: {str(e)}, type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to create document record. Please try again.")

async def update_document_status(job_id: str, status: str, progress: int = None, processing_log: Dict[str, Any] = None):
    """Update document status in database with error handling"""
    try:
        if not job_id or not status:
            logger.warning(f"[v0] Invalid parameters for status update: job_id={job_id}, status={status}")
            return
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            if progress < 0 or progress > 100:
                logger.warning(f"[v0] Invalid progress value: {progress}, clamping to 0-100")
                progress = max(0, min(100, progress))
            
            if not processing_log:
                processing_log = {}
            processing_log["progress"] = progress
            update_data["processing_log"] = processing_log
        
        if status == "formatted":
            update_data["processed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"[v0] Updating document status: job_id={job_id}, status={status}, progress={progress}")
        
        response = supabase.table("documents").update(update_data).eq("id", job_id).execute()
        
        if not response.data:
            logger.warning(f"[v0] Document not found for status update: {job_id}")
        
    except Exception as e:
        logger.error(f"[v0] Error updating document status: {str(e)}, type: {type(e).__name__}")

async def simulate_processing(job_id: str):
    """Simulate document processing with better error handling"""
    try:
        logger.info(f"[v0] Starting processing simulation for job: {job_id}")
        
        await asyncio.sleep(1)
        await update_document_status(job_id, "processing", 25)
        
        await asyncio.sleep(2)
        await update_document_status(job_id, "processing", 75)
        
        await asyncio.sleep(1)
        
        processing_log = {
            "progress": 100,
            "word_count": 1250,
            "headings_count": 8,
            "references_count": 15,
            "processing_time": 3.5
        }
        
        await update_document_status(job_id, "formatted", 100, processing_log)
        logger.info(f"[v0] Processing completed for job: {job_id}")
        
        try:
            doc_response = supabase.table("documents").select("user_id, filename").eq("id", job_id).execute()
            if doc_response.data:
                user_id = doc_response.data[0]["user_id"]
                estimated_storage_mb = processing_log.get("word_count", 1250) * 0.001
                
                await track_document_usage(user_id)
                await track_storage_usage(user_id, estimated_storage_mb)
                logger.info(f"[v0] Usage tracked for user {user_id}")
        except Exception as usage_error:
            logger.error(f"[v0] Non-fatal error tracking usage: {str(usage_error)}")
        
    except Exception as e:
        logger.error(f"[v0] Error in processing simulation: {str(e)}, type: {type(e).__name__}")
        await update_document_status(job_id, "failed", processing_log={"error": str(e)})

async def track_document_usage(user_id: str):
    """Track document usage in subscriptions table"""
    try:
        # Call the database function to increment document usage
        result = supabase.rpc("increment_document_usage", {"p_user_id": user_id}).execute()
        logger.info(f"Document usage incremented for user: {user_id}")
        return result
    except Exception as e:
        logger.error(f"Error tracking document usage: {str(e)}")
        raise

async def track_storage_usage(user_id: str, storage_mb: float):
    """Track storage usage in subscriptions table"""
    try:
        # Convert MB to GB for storage tracking
        storage_gb = storage_mb / 1024
        
        # Call the database function to update storage usage
        result = supabase.rpc("update_storage_usage", {
            "p_user_id": user_id, 
            "storage_gb": storage_gb
        }).execute()
        logger.info(f"Storage usage updated for user: {user_id}, added: {storage_gb}GB")
        return result
    except Exception as e:
        logger.error(f"Error tracking storage usage: {str(e)}")
        raise

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

REQUEST_TIMEOUT = 30  # seconds
SUPABASE_TIMEOUT = 20  # seconds
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

async_client = None

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
