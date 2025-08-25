from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import time
import base64
import os
from datetime import datetime
import asyncio
import jwt
from supabase import create_client, Client
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables validation
class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    PORT = int(os.getenv("PORT", 8000))
    
    def validate(self):
        required_vars = [
            self.SUPABASE_URL, 
            self.SUPABASE_SERVICE_ROLE_KEY, 
            self.SUPABASE_JWT_SECRET, 
            self.SUPABASE_ANON_KEY
        ]
        if not all(required_vars):
            missing = [var for var, val in zip(
                ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_JWT_SECRET', 'SUPABASE_ANON_KEY'],
                required_vars
            ) if not val]
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

config = Config()
config.validate()

# Initialize Supabase client
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)

# Pydantic models
class ProcessDocumentRequest(BaseModel):
    filename: str
    content: str = Field(..., description="base64 encoded content")
    style: str
    englishVariant: str
    reportOnly: bool = False
    includeComments: bool = True
    preserveFormatting: bool = True
    options: Optional[Dict[str, Any]] = None

class JobResponse(BaseModel):
    success: bool
    job_id: str
    message: str

class CreateUploadResponse(JobResponse):
    status: str
    upload_url: str
    upload_token: str
    file_path: str
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
    content: str = Field(..., description="base64 encoded content")
    metadata: Dict[str, Any]

class StyleVariant(BaseModel):
    id: str
    name: str
    description: str

# Constants
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

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Formatly API")
    yield
    logger.info("Shutting down Formatly API")

# Initialize FastAPI app
app = FastAPI(
    title="Formatly API",
    description="Document formatting service API",
    version="1.0.0",
    lifespan=lifespan
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

# Queue and Worker Integration
class QueueService:
    """Service for handling job queue operations"""
    
    @staticmethod
    async def enqueue_job(job_id: str, job_type: str, job_data: Dict[str, Any]):
        """Add job to processing queue"""
        try:
            queue_data = {
                "job_id": job_id,
                "job_type": job_type,
                "job_data": job_data,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "attempts": 0
            }
            
            response = supabase.table("job_queue").insert(queue_data).execute()
            if not response.data:
                raise Exception("Failed to enqueue job")
            
            logger.info(f"Job {job_id} queued for processing")
            return True
            
        except Exception as e:
            logger.error(f"Error enqueueing job: {str(e)}")
            return False

    @staticmethod
    async def update_job_status(job_id: str, status: str, progress: int = None, error: str = None):
        """Update job status in queue"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if progress is not None:
                update_data["progress"] = progress
            
            if error:
                update_data["error_message"] = error
                
            if status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            await supabase.table("job_queue").update(update_data).eq("job_id", job_id).execute()
            
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")

class FormattingService:
    """Service for document formatting operations - interfaces with external formatting models"""
    
    @staticmethod
    async def format_document(job_id: str, document_path: str, style: str, options: Dict[str, Any]):
        """Send document to formatting models service"""
        try:
            # This would interface with the actual Formatting Models service
            formatting_request = {
                "document_path": document_path,
                "style": style,
                "options": options,
                "callback_url": f"/api/formatting/callback/{job_id}"
            }
            
            # In real implementation, this would call the external formatting service
            # For now, simulate the process
            await QueueService.update_job_status(job_id, "processing", 25)
            await asyncio.sleep(2)
            
            await QueueService.update_job_status(job_id, "processing", 75)
            await asyncio.sleep(2)
            
            # Mock successful completion
            result = {
                "formatted_document_path": f"formatted/{job_id}/output.docx",
                "metadata": {
                    "word_count": 1250,
                    "changes_made": 47,
                    "processing_time": 4.2
                }
            }
            
            await QueueService.update_job_status(job_id, "completed", 100)
            return result
            
        except Exception as e:
            logger.error(f"Formatting error for job {job_id}: {str(e)}")
            await QueueService.update_job_status(job_id, "failed", error=str(e))
            return None

# Service classes
class AuthService:
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token with Supabase"""
        if not credentials:
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        try:
            payload = jwt.decode(
                credentials.credentials, 
                config.SUPABASE_JWT_SECRET, 
                algorithms=["HS256"],
                audience="authenticated"
            )
            
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
            # Get user profile
            profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
            if not profile_response.data:
                raise HTTPException(status_code=401, detail="User profile not found")
            
            return {
                "user_id": user_id,
                "email": payload.get("email"),
                "profile": profile_response.data[0]
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise HTTPException(status_code=401, detail="Token verification failed")

class StorageService:
    @staticmethod
    async def generate_signed_upload_url(filename: str, user_id: str) -> Dict[str, str]:
        """Generate signed upload URL for document upload to Supabase Storage"""
        try:
            file_extension = filename.split('.')[-1].lower() if '.' in filename else 'docx'
            unique_filename = f"documents/{user_id}/{int(time.time())}_{uuid.uuid4()}.{file_extension}"
            
            logger.info(f"Generating signed upload URL for: {unique_filename}")
            
            response = supabase.storage.from_("documents").create_signed_upload_url(unique_filename)
            
            # Handle response format variations
            if isinstance(response, dict):
                signed_url = response.get("signedURL") or response.get("signed_url") or response.get("url")
                token = response.get("token", "")
            else:
                signed_url = str(response) if response else None
                token = ""
            
            if not signed_url:
                raise Exception("Failed to generate signed upload URL")
                
            return {
                "upload_url": signed_url,
                "file_path": unique_filename,
                "upload_token": token,
                "upload_headers": {
                    "Authorization": f"Bearer {config.SUPABASE_ANON_KEY}",
                    "x-upsert": "true"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating signed upload URL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")

class DocumentService:
    @staticmethod
    async def create_document_record(
        user_id: str, filename: str, file_path: str, job_id: str, 
        style: str, language_variant: str, options: Dict[str, Any]
    ) -> str:
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

    @staticmethod
    async def update_document_status(
        job_id: str, status: str, progress: int = None, processing_log: Dict[str, Any] = None
    ):
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

    @staticmethod
    async def get_document(job_id: str, user_id: str):
        """Get document by job_id and user_id"""
        response = supabase.table("documents").select("*").eq("id", job_id).eq("user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        return response.data[0]

class ProcessingService:
    @staticmethod
    async def start_document_processing(job_id: str, document_path: str, style: str, options: Dict[str, Any]):
        """Start document processing by enqueueing job"""
        try:
            # Update document status to queued
            await DocumentService.update_document_status(job_id, "queued", 0)
            
            # Enqueue job for worker processing
            job_data = {
                "document_path": document_path,
                "style": style,
                "options": options
            }
            
            success = await QueueService.enqueue_job(job_id, "document_formatting", job_data)
            if not success:
                await DocumentService.update_document_status(job_id, "failed", processing_log={"error": "Failed to enqueue job"})
                return False
            
            # In a real system, a separate worker would pick up this job
            # For demo purposes, we'll simulate immediate processing
            asyncio.create_task(ProcessingService._simulate_worker_processing(job_id, document_path, style, options))
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting document processing: {str(e)}")
            await DocumentService.update_document_status(job_id, "failed", processing_log={"error": str(e)})
            return False

    @staticmethod
    async def _simulate_worker_processing(job_id: str, document_path: str, style: str, options: Dict[str, Any]):
        """Simulate worker processing (in real system, this would be a separate service)"""
        try:
            # Update status to processing
            await DocumentService.update_document_status(job_id, "processing", 10)
            await QueueService.update_job_status(job_id, "processing", 10)
            
            # Call formatting service
            result = await FormattingService.format_document(job_id, document_path, style, options)
            
            if result:
                processing_log = {
                    "progress": 100,
                    "word_count": result["metadata"]["word_count"],
                    "changes_made": result["metadata"]["changes_made"],
                    "processing_time": result["metadata"]["processing_time"],
                    "formatted_document_path": result["formatted_document_path"]
                }
                
                await DocumentService.update_document_status(job_id, "formatted", 100, processing_log)
            else:
                await DocumentService.update_document_status(job_id, "failed", processing_log={"error": "Formatting service failed"})
                
        except Exception as e:
            logger.error(f"Error in worker processing: {str(e)}")
            await DocumentService.update_document_status(job_id, "failed", processing_log={"error": str(e)})
            await QueueService.update_job_status(job_id, "failed", error=str(e))

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

@app.post("/api/documents/create-upload", response_model=CreateUploadResponse)
async def create_upload_url(
    filename: str = Form(...),
    style: str = Form(...),
    englishVariant: str = Form(...),
    reportOnly: bool = Form(False),
    includeComments: bool = Form(True),
    preserveFormatting: bool = Form(True),
    user: dict = Depends(AuthService.verify_token)
):
    """Generate signed upload URL and create document record"""
    try:
        job_id = str(uuid.uuid4())
        
        # Generate upload URL
        upload_info = await StorageService.generate_signed_upload_url(filename, user["user_id"])
        
        # Create document record
        options = {
            "reportOnly": reportOnly,
            "includeComments": includeComments,
            "preserveFormatting": preserveFormatting
        }
        
        await DocumentService.create_document_record(
            user["user_id"], filename, upload_info["file_path"], 
            job_id, style, englishVariant, options
        )
        
        return CreateUploadResponse(
            success=True,
            job_id=job_id,
            status="draft",
            upload_url=upload_info["upload_url"],
            upload_token=upload_info["upload_token"],
            file_path=upload_info["file_path"],
            upload_headers=upload_info["upload_headers"],
            message=f"Upload URL created. Ready for {style} formatting."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create upload URL error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create upload URL: {str(e)}")

@app.post("/api/documents/upload-complete")
async def upload_complete_webhook(
    webhook_data: WebhookUploadComplete,
    user: dict = Depends(AuthService.verify_token)
):
    """Webhook called after successful file upload"""
    try:
        document = await DocumentService.get_document(webhook_data.job_id, user["user_id"])
        
        if webhook_data.success:
            await DocumentService.update_document_status(webhook_data.job_id, "uploaded", 5)
            
            # Start processing via queue system
            success = await ProcessingService.start_document_processing(
                webhook_data.job_id, 
                webhook_data.file_path, 
                document["style_applied"], 
                document["formatting_options"]
            )
            
            if success:
                message = "Upload confirmed. Document queued for processing."
            else:
                message = "Upload confirmed but processing failed to start."
        else:
            await DocumentService.update_document_status(
                webhook_data.job_id, "failed", 
                processing_log={"error": "File upload failed"}
            )
            message = "Upload failed. Please try again."
        
        return {
            "success": webhook_data.success,
            "message": message,
            "job_id": webhook_data.job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload complete webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.post("/api/documents/process", response_model=JobResponse)
async def process_document(
    request: ProcessDocumentRequest,
    user: dict = Depends(AuthService.verify_token)
):
    """Process document with formatting (legacy endpoint)"""
    try:
        job_id = str(uuid.uuid4())
        
        options = {
            "reportOnly": request.reportOnly,
            "includeComments": request.includeComments,
            "preserveFormatting": request.preserveFormatting,
            "content": request.content
        }
        
        await DocumentService.create_document_record(
            user["user_id"], request.filename, f"legacy/{job_id}",
            job_id, request.style, request.englishVariant, options
        )
        
        # Use queue-based processing for consistency
        await ProcessingService.start_document_processing(
            job_id, f"legacy/{job_id}", request.style, options
        )
        
        return JobResponse(
            success=True,
            job_id=job_id,
            message=f"Document queued for {request.style} formatting"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/documents/status/{job_id}", response_model=DocumentStatusResponse)
async def get_document_status(job_id: str, user: dict = Depends(AuthService.verify_token)):
    """Get document processing status"""
    try:
        document = await DocumentService.get_document(job_id, user["user_id"])
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

@app.get("/api/documents/download/{job_id}", response_model=FormattedDocumentResponse)
async def download_document(job_id: str, user: dict = Depends(AuthService.verify_token)):
    """Download formatted document"""
    try:
        document = await DocumentService.get_document(job_id, user["user_id"])
        
        if document["status"] != "formatted":
            raise HTTPException(status_code=400, detail="Document not ready for download")
        
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

@app.get("/api/formatting/styles", response_model=List[StyleVariant])
async def get_formatting_styles():
    """Get available formatting styles"""
    try:
        response = supabase.table("active_formatting_styles").select("*").order("sort_order").execute()
        
        if response.data:
            return [
                StyleVariant(
                    id=style["code"],
                    name=style["name"],
                    description=style["description"]
                )
                for style in response.data
            ]
        else:
            return [StyleVariant(**style) for style in FORMATTING_STYLES]
            
    except Exception as e:
        logger.error(f"Error fetching formatting styles: {str(e)}")
        return [StyleVariant(**style) for style in FORMATTING_STYLES]

@app.get("/api/formatting/variants", response_model=List[StyleVariant])
async def get_english_variants():
    """Get available English variants"""
    try:
        response = supabase.table("active_english_variants").select("*").order("sort_order").execute()
        
        if response.data:
            return [
                StyleVariant(
                    id=variant["code"],
                    name=variant["name"],
                    description=variant["description"]
                )
                for variant in response.data
            ]
        else:
            return [StyleVariant(**variant) for variant in ENGLISH_VARIANTS]
            
    except Exception as e:
        logger.error(f"Error fetching English variants: {str(e)}")
        return [StyleVariant(**variant) for variant in ENGLISH_VARIANTS]

@app.post("/api/formatting/callback/{job_id}")
async def formatting_callback(job_id: str, callback_data: dict):
    """Callback endpoint for formatting service to report completion"""
    try:
        if callback_data.get("success"):
            processing_log = {
                "progress": 100,
                "formatted_document_path": callback_data.get("output_path"),
                "metadata": callback_data.get("metadata", {})
            }
            await DocumentService.update_document_status(job_id, "formatted", 100, processing_log)
        else:
            error_msg = callback_data.get("error", "Unknown formatting error")
            await DocumentService.update_document_status(job_id, "failed", processing_log={"error": error_msg})
        
        return {"success": True, "message": "Callback processed"}
        
    except Exception as e:
        logger.error(f"Formatting callback error: {str(e)}")
        raise HTTPException(status_code=500, detail="Callback processing failed")

@app.get("/api/queue/stats")
async def get_queue_stats():
    """Get queue statistics for monitoring"""
    try:
        response = supabase.table("job_queue").select("status", count="exact").execute()
        
        stats = {}
        for row in response.data:
            status = row["status"]
            stats[status] = stats.get(status, 0) + 1
            
        return {
            "queue_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Queue stats error: {str(e)}")
        return {"queue_stats": {}, "error": str(e)}

# Removed legacy upload endpoint and test endpoints for cleaner production code

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
