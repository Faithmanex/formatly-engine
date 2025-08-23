from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import time
import base64
import json
import os
import io
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

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_JWT_SECRET]):
    raise ValueError("Missing required Supabase environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Security
security = HTTPBearer(auto_error=False)

# Enhanced Pydantic models matching React component expectations
class UploadRequest(BaseModel):
    filename: str = Field(..., description="Original filename")
    fileSize: int = Field(..., ge=1, le=10*1024*1024, description="File size in bytes (max 10MB)")
    contentType: str = Field(..., description="MIME type of the file")

class UploadResponse(BaseModel):
    uploadUrl: str = Field(..., description="Pre-signed URL for file upload")
    jobId: str = Field(..., description="Unique job identifier")
    filePath: str = Field(..., description="Storage path for the file")

class ProcessRequest(BaseModel):
    jobId: str = Field(..., description="Job ID from upload response")
    filename: str = Field(..., description="Original filename")
    style: str = Field(..., description="Formatting style (APA, MLA, etc.)")
    englishVariant: str = Field(..., description="English variant (US, UK, etc.)")
    reportOnly: bool = Field(default=False, description="Generate report only")
    includeComments: bool = Field(default=False, description="Include comments in processing")
    preserveFormatting: bool = Field(default=False, description="Preserve original formatting")

class ProcessResponse(BaseModel):
    success: bool
    jobId: str
    status: str
    message: str

class StatusResponse(BaseModel):
    status: str = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    downloadUrl: Optional[str] = Field(None, description="Download URL when completed")
    error: Optional[str] = Field(None, description="Error message if failed")

# Status mapping between database and frontend
STATUS_MAPPING = {
    # Database -> Frontend
    "draft": "pending",
    "queued": "queued", 
    "processing": "processing",
    "formatted": "completed",
    "failed": "failed",
    "error": "error"
}

REVERSE_STATUS_MAPPING = {v: k for k, v in STATUS_MAPPING.items()}

# Helper functions
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Enhanced token verification with better error handling"""
    if not credentials:
        raise HTTPException(
            status_code=401, 
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        token = credentials.credentials
        
        # Validate token format
        if not token or len(token.split('.')) != 3:
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        # Decode JWT token
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=["HS256"],
            audience="authenticated"
        )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        
        # Validate UUID format
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid user ID format")
        
        # Get user profile from database with timeout
        try:
            profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
        except Exception as e:
            logger.error(f"Database error during token verification: {str(e)}")
            raise HTTPException(status_code=503, detail="Authentication service temporarily unavailable")
        
        if not profile_response.data:
            raise HTTPException(status_code=401, detail="User profile not found")
        
        profile = profile_response.data[0]
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "profile": profile
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, 
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected token verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Token verification failed")

async def generate_signed_upload_url(filename: str, user_id: str, file_size: int) -> Dict[str, str]:
    """Generate signed upload URL with enhanced validation"""
    try:
        # Validate filename
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename")
        
        # Extract and validate file extension
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'bin'
        allowed_extensions = ['doc', 'docx', 'pdf', 'txt', 'rtf']
        
        if file_extension not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create unique file path with timestamp for collision avoidance
        timestamp = int(time.time())
        unique_filename = f"{user_id}/{timestamp}_{uuid.uuid4()}.{file_extension}"
        
        # Generate signed upload URL with expiration
        response = supabase.storage.from_("documents").create_signed_upload_url(
            unique_filename,
            expires_in=3600  # 1 hour expiration
        )
        
        if not response or not response.get("signedURL"):
            logger.error(f"Failed to generate signed upload URL: {response}")
            raise Exception("Storage service returned invalid response")
        
        return {
            "upload_url": response["signedURL"],
            "file_path": unique_filename,
            "upload_token": response.get("token", "")
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating signed upload URL: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

async def create_document_record(
    job_id: str,
    user_id: str, 
    filename: str, 
    file_path: str, 
    file_size: int,
    style: str, 
    language_variant: str, 
    options: Dict[str, Any]
) -> str:
    """Create document record with enhanced validation"""
    try:
        # Validate inputs
        if not all([job_id, user_id, filename, file_path]):
            raise ValueError("Missing required document parameters")
        
        # Validate file size
        if file_size <= 0 or file_size > 10 * 1024 * 1024:  # 10MB max
            raise ValueError("Invalid file size")
        
        # Extract file type safely
        file_type = filename.split('.')[-1].lower() if '.' in filename else "unknown"
        
        document_data = {
            "id": job_id,
            "user_id": user_id,
            "filename": filename,
            "original_filename": filename,
            "file_size": file_size,
            "file_type": file_type,
            "status": "draft",  # Initial status
            "style_applied": style,
            "language_variant": language_variant,
            "report_only": options.get("reportOnly", False),
            "storage_location": file_path,
            "formatting_options": {
                "includeComments": options.get("includeComments", False),
                "preserveFormatting": options.get("preserveFormatting", False),
                "reportOnly": options.get("reportOnly", False),
                "style": style,
                "englishVariant": language_variant
            },
            "processing_log": {"created_at": datetime.utcnow().isoformat()},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table("documents").insert(document_data).execute()
        
        if not response.data:
            logger.error(f"Failed to create document record: {response}")
            raise Exception("Database insertion failed")
        
        logger.info(f"Document record created successfully: {job_id}")
        return job_id
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating document record: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create document record")

async def update_document_status(
    job_id: str, 
    status: str, 
    progress: Optional[int] = None, 
    processing_log: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
):
    """Update document status with comprehensive logging"""
    try:
        # Validate status
        valid_statuses = ["draft", "queued", "processing", "formatted", "failed", "error"]
        if status not in valid_statuses:
            logger.warning(f"Invalid status: {status}, defaulting to 'error'")
            status = "error"
        
        # Prepare update data
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Handle processing log updates
        if processing_log or progress is not None or error_message:
            # Get existing processing log
            existing_response = supabase.table("documents").select("processing_log").eq("id", job_id).execute()
            existing_log = {}
            
            if existing_response.data and existing_response.data[0]:
                existing_log = existing_response.data[0].get("processing_log") or {}
            
            # Update log with new information
            if progress is not None:
                existing_log["progress"] = progress
                existing_log["last_progress_update"] = datetime.utcnow().isoformat()
            
            if error_message:
                existing_log["error"] = error_message
                existing_log["error_time"] = datetime.utcnow().isoformat()
            
            if processing_log:
                existing_log.update(processing_log)
            
            update_data["processing_log"] = existing_log
        
        # Set processed_at timestamp for completion
        if status in ["formatted", "completed"]:
            update_data["processed_at"] = datetime.utcnow().isoformat()
        
        # Execute update
        response = supabase.table("documents").update(update_data).eq("id", job_id).execute()
        
        if not response.data:
            logger.warning(f"No document found with ID: {job_id}")
        else:
            logger.info(f"Document {job_id} status updated to: {status}")
        
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")
        # Don't raise exception here to avoid breaking the processing flow

async def simulate_processing(job_id: str):
    """Enhanced processing simulation with realistic timing and error handling"""
    try:
        logger.info(f"Starting processing simulation for job: {job_id}")
        
        # Stage 1: Initial processing
        await update_document_status(job_id, "processing", 10)
        await asyncio.sleep(2)
        
        # Stage 2: Analysis phase
        await update_document_status(job_id, "processing", 35, {
            "stage": "analyzing_document",
            "stage_start": datetime.utcnow().isoformat()
        })
        await asyncio.sleep(3)
        
        # Stage 3: Formatting phase
        await update_document_status(job_id, "processing", 65, {
            "stage": "applying_formatting",
            "stage_start": datetime.utcnow().isoformat()
        })
        await asyncio.sleep(2)
        
        # Stage 4: Finalization
        await update_document_status(job_id, "processing", 90, {
            "stage": "finalizing",
            "stage_start": datetime.utcnow().isoformat()
        })
        await asyncio.sleep(1)
        
        # Completion with metrics
        processing_log = {
            "progress": 100,
            "word_count": 1250,
            "headings_count": 8,
            "references_count": 15,
            "processing_time": 8.0,  # Total time in seconds
            "stage": "completed",
            "completion_time": datetime.utcnow().isoformat()
        }
        
        await update_document_status(job_id, "formatted", 100, processing_log)
        logger.info(f"Processing completed successfully for job: {job_id}")
        
    except Exception as e:
        logger.error(f"Error in processing simulation for job {job_id}: {str(e)}")
        await update_document_status(
            job_id, 
            "failed", 
            error_message=f"Processing failed: {str(e)}"
        )

def map_status_for_frontend(db_status: str) -> str:
    """Map database status to frontend expected values"""
    return STATUS_MAPPING.get(db_status, db_status)

def validate_file_request(filename: str, file_size: int, content_type: str) -> None:
    """Validate file upload request parameters"""
    # Check filename
    if not filename or len(filename.strip()) == 0:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if len(filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long (max 255 characters)")
    
    # Check file size
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="File size must be greater than 0")
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    # Check file extension
    if '.' not in filename:
        raise HTTPException(status_code=400, detail="File must have an extension")
    
    file_extension = filename.split('.')[-1].lower()
    allowed_extensions = ['doc', 'docx', 'pdf', 'txt', 'rtf']
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate content type if provided
    if content_type:
        valid_content_types = {
            'doc': ['application/msword'],
            'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            'pdf': ['application/pdf'],
            'txt': ['text/plain'],
            'rtf': ['application/rtf', 'text/rtf']
        }
        
        expected_types = valid_content_types.get(file_extension, [])
        if expected_types and content_type not in expected_types:
            logger.warning(f"Content type mismatch: expected {expected_types}, got {content_type}")

# Enhanced API Routes

@app.get("/")
async def root():
    return {
        "message": "Formatly API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "upload": "/api/documents/upload",
            "process": "/api/documents/process", 
            "status": "/api/documents/status/{job_id}",
            "download": "/api/documents/download/{job_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with dependency verification"""
    try:
        # Test database connection
        test_response = supabase.table("documents").select("id").limit(1).execute()
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_healthy = False
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "healthy" if db_healthy else "unhealthy",
            "storage": "healthy"  # Add actual storage check if needed
        }
    }

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: UploadRequest,
    user: dict = Depends(verify_token)
) -> UploadResponse:
    """Generate pre-signed upload URL - FIXED TO MATCH REACT COMPONENT"""
    try:
        logger.info(f"Upload request for user {user['user_id'][-8:]}: {request.filename}")
        
        # Validate request parameters
        validate_file_request(request.filename, request.fileSize, request.contentType)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Generate signed upload URL
        upload_info = await generate_signed_upload_url(
            request.filename, 
            user["user_id"],
            request.fileSize
        )
        
        logger.info(f"Generated upload URL for job {job_id}")
        
        # Return structure matching React component expectations
        return UploadResponse(
            uploadUrl=upload_info["upload_url"],
            jobId=job_id,
            filePath=upload_info["file_path"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create upload URL")

@app.post("/api/documents/process", response_model=ProcessResponse)
async def process_document(
    request: ProcessRequest,
    user: dict = Depends(verify_token)
) -> ProcessResponse:
    """Process uploaded document - FIXED TO MATCH REACT COMPONENT"""
    try:
        logger.info(f"Process request for job {request.jobId}: {request.filename}")
        
        # Validate job ID format
        try:
            uuid.UUID(request.jobId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        # Validate style and variant
        if not request.style or len(request.style.strip()) == 0:
            raise HTTPException(status_code=400, detail="Style is required")
        
        if not request.englishVariant or len(request.englishVariant.strip()) == 0:
            raise HTTPException(status_code=400, detail="English variant is required")
        
        # Prepare formatting options
        options = {
            "reportOnly": request.reportOnly,
            "includeComments": request.includeComments,
            "preserveFormatting": request.preserveFormatting
        }
        
        # Create document record with the provided job ID
        await create_document_record(
            job_id=request.jobId,
            user_id=user["user_id"],
            filename=request.filename,
            file_path=f"documents/{request.jobId}",  # Storage path
            file_size=0,  # Will be updated after upload verification
            style=request.style,
            language_variant=request.englishVariant,
            options=options
        )
        
        # Update status to queued
        await update_document_status(request.jobId, "queued", 0)
        
        # Start background processing
        asyncio.create_task(simulate_processing(request.jobId))
        
        logger.info(f"Started processing for job {request.jobId}")
        
        return ProcessResponse(
            success=True,
            jobId=request.jobId,
            status="queued",
            message=f"Document queued for {request.style} formatting"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@app.get("/api/documents/status/{job_id}", response_model=StatusResponse)
async def get_document_status(
    job_id: str, 
    user: dict = Depends(verify_token)
) -> StatusResponse:
    """Get document status - FIXED TO MATCH REACT EXPECTATIONS"""
    try:
        # Validate job ID format
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        # Get document from database
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = response.data[0]
        processing_log = document.get("processing_log") or {}
        
        # Map status to frontend expected value
        frontend_status = map_status_for_frontend(document["status"])
        
        # Get progress
        progress = processing_log.get("progress", 0)
        
        # Generate download URL if completed
        download_url = None
        if frontend_status == "completed":
            download_url = f"/api/documents/download/{job_id}"
        
        # Get error message if failed
        error_message = None
        if frontend_status in ["failed", "error"]:
            error_message = processing_log.get("error", "Processing failed")
        
        return StatusResponse(
            status=frontend_status,
            progress=progress,
            downloadUrl=download_url,
            error=error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document status")

@app.get("/api/documents/download/{job_id}")
async def download_document(
    job_id: str, 
    user: dict = Depends(verify_token)
):
    """Download formatted document - RETURNS BINARY STREAM"""
    try:
        # Validate job ID
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        # Get document from database
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = response.data[0]
        
        # Check if document is ready for download
        if document["status"] != "formatted":
            frontend_status = map_status_for_frontend(document["status"])
            raise HTTPException(
                status_code=400, 
                detail=f"Document not ready for download. Current status: {frontend_status}"
            )
        
        # Generate mock formatted content
        processing_log = document.get("processing_log") or {}
        
        formatted_content = f"""FORMATTED DOCUMENT
==============================
Original File: {document['filename']}
Style Applied: {document['style_applied']}
Language Variant: {document['language_variant']}
Processing Time: {processing_log.get('processing_time', 0)} seconds

Document Statistics:
- Word Count: {processing_log.get('word_count', 0)}
- Headings: {processing_log.get('headings_count', 0)}
- References: {processing_log.get('references_count', 0)}

Formatting Options:
- Report Only: {document.get('report_only', False)}
- Include Comments: {document.get('formatting_options', {}).get('includeComments', False)}
- Preserve Formatting: {document.get('formatting_options', {}).get('preserveFormatting', False)}

This is a mock formatted document for demonstration purposes.
In production, this would contain the actual formatted content.
"""
        
        # Convert to bytes
        content_bytes = formatted_content.encode('utf-8')
        
        # Generate appropriate filename
        original_name = document['filename']
        name_without_ext = '.'.join(original_name.split('.')[:-1]) if '.' in original_name else original_name
        formatted_filename = f"formatted_{name_without_ext}.txt"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(content_bytes),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=\"{formatted_filename}\"",
                "Content-Length": str(len(content_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download document")

@app.get("/api/formatting/styles")
async def get_formatting_styles():
    """Get available formatting styles"""
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
            # Fallback to default styles
            return [
                {"id": "APA", "name": "APA Style", "description": "American Psychological Association"},
                {"id": "MLA", "name": "MLA Style", "description": "Modern Language Association"},
                {"id": "Chicago", "name": "Chicago Style", "description": "Chicago Manual of Style"},
                {"id": "IEEE", "name": "IEEE Style", "description": "Institute of Electrical and Electronics Engineers"},
            ]
            
    except Exception as e:
        logger.error(f"Error fetching formatting styles: {str(e)}")
        return [{"id": "APA", "name": "APA Style", "description": "American Psychological Association"}]

@app.get("/api/formatting/variants")
async def get_english_variants():
    """Get available English variants"""
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
            # Fallback to default variants
            return [
                {"id": "US", "name": "US English", "description": "American English"},
                {"id": "UK", "name": "UK English", "description": "British English"},
                {"id": "CA", "name": "Canadian English", "description": "Canadian English"},
                {"id": "AU", "name": "Australian English", "description": "Australian English"},
            ]
            
    except Exception as e:
        logger.error(f"Error fetching English variants: {str(e)}")
        return [{"id": "US", "name": "US English", "description": "American English"}]

# Admin/Debug endpoints
@app.get("/api/admin/jobs")
async def list_user_jobs(user: dict = Depends(verify_token)):
    """List all jobs for the authenticated user"""
    try:
        response = supabase.table("documents").select("*").eq("user_id", user["user_id"]).order("created_at", desc=True).execute()
        
        return {
            "jobs": response.data or [],
            "total": len(response.data or [])
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")

@app.delete("/api/admin/jobs/{job_id}")
async def delete_job(job_id: str, user: dict = Depends(verify_token)):
    """Delete a job (admin/debug endpoint)"""
    try:
        uuid.UUID(job_id)  # Validate format
        
        response = supabase.table("documents").delete().eq("id", job_id).eq("user_id", user["user_id"]).execute()
        
        if response.data:
            return {"success": True, "message": "Job deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete job")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Formatly API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
