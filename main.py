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
    trackedChanges: bool = False
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
    content: str  # base64 encoded - neat copy
    tracked_changes_content: Optional[str] = None  # base64 encoded - version with tracked changes
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
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'docx'
        timestamp = int(time.time())
        unique_filename = f"documents/{user_id}/{timestamp}_{uuid.uuid4()}.{file_extension}"
        
        logger.info(f"Generating signed upload URL for file path: {unique_filename}")
        
        storage_bucket = "documents"
        try:
            # Use the correct Supabase storage method
            response = supabase.storage.from_(storage_bucket).create_signed_upload_url(unique_filename)
            logger.info(f"Supabase storage response: {response}")
            
            # Handle different response formats from Supabase
            if isinstance(response, dict):
                signed_url = response.get("signedURL") or response.get("signed_url") or response.get("url")
                token = response.get("token", "")
            else:
                # If response is a string, it might be the URL directly
                signed_url = str(response) if response else None
                token = ""
            
            if not signed_url:
                logger.error(f"No signed URL in response: {response}")
                raise Exception("Failed to generate signed upload URL - no signedURL returned")
                
        except Exception as storage_error:
            logger.error(f"Supabase storage error: {str(storage_error)}")
            # Try alternative approach with simpler client
            try:
                # Create a simpler client for storage operations
                storage_client = supabase.storage.from_(storage_bucket)
                response = storage_client.create_signed_upload_url(unique_filename)
                logger.info(f"Alternative storage response: {response}")
                
                if isinstance(response, dict):
                    signed_url = response.get("signedURL") or response.get("signed_url") or response.get("url")
                    token = response.get("token", "")
                else:
                    signed_url = str(response) if response else None
                    token = ""
                    
                if not signed_url:
                    raise Exception("Alternative method also failed to generate signed URL")
                    
            except Exception as alt_error:
                logger.error(f"Alternative storage method failed: {str(alt_error)}")
                raise Exception(f"Both storage methods failed: {str(storage_error)} | {str(alt_error)}")
        
        upload_headers = {
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "x-upsert": "true"
        }
        
        logger.info(f"Successfully generated signed URL for: {unique_filename}")
        
        return {
            "upload_url": signed_url,
            "file_path": unique_filename,
            "upload_token": token,
            "upload_headers": upload_headers
        }
        
    except Exception as e:
        logger.error(f"Error generating signed upload URL: {str(e)}")
        logger.error(f"Supabase URL: {SUPABASE_URL}")
        logger.error(f"Storage bucket: documents")
        logger.error(f"User ID: {user_id}")
        logger.error(f"Filename: {filename}")
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")

async def generate_presigned_upload_url(filename: str, user_id: str) -> Dict[str, str]:
    """Legacy function - redirects to corrected version"""
    return await generate_signed_upload_url(filename, user_id)

async def create_document_record(user_id: str, filename: str, file_path: str, job_id: str, style: str, language_variant: str, options: Dict[str, Any], file_size: Optional[int] = None) -> str:
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
            "tracked_changes": options.get("trackedChanges", False),
            "file_type": filename.split('.')[-1].lower() if '.' in filename else "docx",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if file_size is not None:
            if file_size < 0:
                raise HTTPException(status_code=400, detail="File size must be non-negative")
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 100MB")
            document_data["file_size"] = file_size
            logger.info(f"Document record created with file size: {file_size} bytes")
        
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
            "processing_time": 3.5
        }
        
        await update_document_status(job_id, "formatted", 100, processing_log)
        
        try:
            # Get document details to find user_id
            doc_response = supabase.table("documents").select("user_id, filename").eq("id", job_id).execute()
            if doc_response.data:
                user_id = doc_response.data[0]["user_id"]
                filename = doc_response.data[0]["filename"]
                
                # Calculate storage used (mock calculation based on file processing)
                estimated_storage_mb = processing_log.get("word_count", 1000) * 0.001  # Rough estimate
                
                # Call usage tracking functions
                await track_document_usage(user_id)
                await track_storage_usage(user_id, estimated_storage_mb)
                
                logger.info(f"Usage tracked for user {user_id}: document processed, storage: {estimated_storage_mb}MB")
            else:
                logger.warning(f"Could not find document record for job_id: {job_id}")
                
        except Exception as usage_error:
            logger.error(f"Error tracking usage for job {job_id}: {str(usage_error)}")
            # Don't fail the entire process if usage tracking fails
        
    except Exception as e:
        logger.error(f"Error in processing simulation: {str(e)}")
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
            "p_storage_gb": storage_gb
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

@app.post("/api/documents/create-upload")
async def create_upload_url(
    filename: str = Form(...),
    style: str = Form(...),
    englishVariant: str = Form(...),
    trackedChanges: bool = Form(False),
    file_size: Optional[int] = Form(None),
    user: dict = Depends(verify_token)
) -> CreateUploadResponse:
    """NEW ENDPOINT: Generate signed upload URL and create document record - PROPER FLOW"""
    try:
        job_id = str(uuid.uuid4())
        
        if file_size is not None:
            if file_size < 0:
                raise HTTPException(status_code=400, detail="File size must be non-negative")
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 100MB")
            logger.info(f"Creating upload for file: {filename}, size: {file_size} bytes")
        
        # STEP 1: Generate signed upload URL first
        upload_info = await generate_signed_upload_url(filename, user["user_id"])
        
        # STEP 2: Create document record with DRAFT status
        options = {
            "trackedChanges": trackedChanges
        }
        
        await create_document_record(
            user["user_id"], 
            filename, 
            upload_info["file_path"], 
            job_id, 
            style, 
            englishVariant, 
            options,
            file_size
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
    trackedChanges: bool = Form(False),
    file_size: Optional[int] = Form(None),
    user: dict = Depends(verify_token)
):
    """EXISTING ENDPOINT: Generate presigned upload URL and create document record - KEPT FOR COMPATIBILITY"""
    try:
        job_id = str(uuid.uuid4())
        
        if file_size is not None:
            if file_size < 0:
                raise HTTPException(status_code=400, detail="File size must be non-negative")
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 100MB")
            logger.info(f"Legacy upload for file: {filename}, size: {file_size} bytes")
        
        # FIXED: Use corrected upload URL generation
        upload_info = await generate_signed_upload_url(filename, user["user_id"])
        
        options = {
            "trackedChanges": trackedChanges
        }
        
        # Create document record in database
        await create_document_record(
            user["user_id"], 
            filename, 
            upload_info["file_path"], 
            job_id, 
            style, 
            englishVariant,
            options,
            file_size
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
        
        options = {
            "trackedChanges": request.trackedChanges,
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
    try:
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        document = response.data[0]
        
        if document["status"] != "formatted":
            raise HTTPException(status_code=400, detail="Document not ready for download")
        
        processing_log = document.get("processing_log") or {}
        formatting_options = document.get("formatting_options") or {}
        
        mock_content = f"FORMATTED DOCUMENT: {document['filename']}\nStyle: {document['style_applied']}\nVariant: {document['language_variant']}"
        formatted_content = base64.b64encode(mock_content.encode()).decode()
        
        tracked_changes_content = None
        # Check both the column and the options for backward compatibility
        tracked_changes_enabled = document.get("tracked_changes") or formatting_options.get("trackedChanges", False)
        
        if tracked_changes_enabled:
            tracked_content = f"TRACKED CHANGES VERSION: {document['filename']}\nStyle: {document['style_applied']}\nVariant: {document['language_variant']}\n\n[CHANGES TRACKED:]\n- Paragraph formatting adjusted\n- Heading styles applied\n- Spacing normalized\n- References formatted"
            tracked_changes_content = base64.b64encode(tracked_content.encode()).decode()
        
        metadata = {
            "word_count": processing_log.get("word_count", 1250),
            "style_applied": document["style_applied"],
            "processing_time": processing_log.get("processing_time", 3.5),
            "tracked_changes_enabled": tracked_changes_enabled
        }
        
        return FormattedDocumentResponse(
            success=True,
            filename=f"formatted_{document['filename']}",
            content=formatted_content,
            tracked_changes_content=tracked_changes_content,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/api/formatting/styles")
async def get_formatting_styles():
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
            return FORMATTING_STYLES
            
    except Exception as e:
        logger.error(f"Error fetching formatting styles: {str(e)}")
        return FORMATTING_STYLES

@app.get("/api/formatting/variants")
async def get_english_variants():
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
            return ENGLISH_VARIANTS
            
    except Exception as e:
        logger.error(f"Error fetching English variants: {str(e)}")
        return ENGLISH_VARIANTS

# Additional utility endpoints for testing

@app.get("/api/jobs")
async def list_jobs():
    return {
        "jobs": list(jobs_storage.values()),
        "total": len(jobs_storage)
    }

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id in jobs_storage:
        del jobs_storage[job_id]
        return {"success": True, "message": "Job deleted"}
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/files")
async def list_files():
    return {
        "files": list(files_storage.values()),
        "total": len(files_storage)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
