# Formatly FastAPI Backend

This is a FastAPI backend for the Formatly document formatting application with full Supabase integration. It provides real JWT authentication, presigned URL generation for secure file uploads, and database integration for job tracking.

## 🚀 Quick Start

### Environment Variables

Before running the backend, set these required environment variables:

\`\`\`env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
PORT=8000
\`\`\`

### Local Development

1. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Set environment variables:**
   \`\`\`bash
   export SUPABASE_URL="your_supabase_url"
   export SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"
   export SUPABASE_JWT_SECRET="your_jwt_secret"
   \`\`\`

3. **Run the server:**
   \`\`\`bash
   python main.py
   \`\`\`

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - OpenAPI spec: http://localhost:8000/openapi.json

### Deploy to Render

1. **Create a new Web Service on Render**
2. **Connect your repository**
3. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
   - **Environment:** Python 3.11

4. **Set environment variables:**
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key
   - `SUPABASE_JWT_SECRET`: Your Supabase JWT secret
   - `PORT`: 8000 (automatically set by Render)

5. **Deploy and get your service URL**

## 📋 API Endpoints

### Core Document Processing
- `POST /api/documents/upload` - Generate presigned upload URL and create document record
- `POST /api/documents/process` - Start document formatting (legacy endpoint)
- `GET /api/documents/status/{job_id}` - Check processing status from database
- `GET /api/documents/download/{job_id}` - Download formatted document

### Configuration
- `GET /api/formatting/styles` - Get available formatting styles from database
- `GET /api/formatting/variants` - Get English variants from database

### Testing Utilities
- `GET /api/jobs` - List all processing jobs (in-memory)
- `DELETE /api/jobs/{job_id}` - Delete a job (in-memory)
- `GET /api/files` - List uploaded files (in-memory)

## 🔐 Authentication

The backend now implements **real Supabase JWT authentication**:

- All protected endpoints require a valid JWT token in the Authorization header
- Tokens are verified against your Supabase JWT secret
- User profiles are fetched from the `profiles` table
- Invalid or expired tokens return 401 Unauthorized

### Frontend Integration

Update your frontend's environment variables:

\`\`\`env
FASTAPI_BASE_URL=https://your-render-service.onrender.com
FASTAPI_TIMEOUT=30000
\`\`\`

Ensure your frontend sends the JWT token in requests:
\`\`\`javascript
headers: {
  'Authorization': `Bearer ${supabaseToken}`
}
\`\`\`

## 🗄️ Database Integration

The backend integrates with your Supabase database:

### Tables Used:
- `documents` - Stores document metadata and processing status
- `profiles` - User profile information for authentication
- `active_formatting_styles` - Available formatting styles
- `active_english_variants` - Available English variants

### Document Processing Flow:
1. **Upload Request** → Generate presigned URL + Create document record
2. **File Upload** → Client uploads directly to Supabase Storage
3. **Processing** → Update document status in database
4. **Completion** → Store results and metadata in database

## 📁 File Storage

Documents are stored in Supabase Storage:
- **Bucket**: `documents`
- **Path Structure**: `{user_id}/{uuid}.{extension}`
- **Security**: Presigned URLs for secure uploads
- **Access**: Authenticated users can only access their own files

## 🧪 Testing

### Health Check
\`\`\`bash
curl https://your-service.onrender.com/health
\`\`\`

### Upload Test (with Authentication)
\`\`\`bash
curl -X POST "https://your-service.onrender.com/api/documents/upload" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "filename=test.docx" \
  -F "style=apa" \
  -F "englishVariant=us"
\`\`\`

### Check Status
\`\`\`bash
curl -X GET "https://your-service.onrender.com/api/documents/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
\`\`\`

## 🔄 Current Implementation

### ✅ Implemented Features:
- **Real JWT Authentication** with Supabase
- **Presigned URL Generation** for secure file uploads
- **Database Integration** for job tracking and user management
- **Document Status Tracking** in real-time
- **User Profile Integration** from Supabase
- **Formatting Styles/Variants** from database

### 🚧 Mock Features (To Be Implemented):
- **Actual Document Processing** - Currently simulated with 4-second delay
- **Real Document Formatting** - Returns mock formatted content
- **File Processing Logic** - Actual Word document manipulation

## 🚀 Production Considerations

The backend is production-ready except for document processing:

### ✅ Production Ready:
- JWT authentication and authorization
- Database integration with proper error handling
- Secure file upload with presigned URLs
- Proper logging and error handling
- CORS configuration
- Environment variable configuration

### 🔧 Still Needed:
1. **Implement actual document processing** logic
2. **Add document parsing** (python-docx, PyPDF2, etc.)
3. **Implement formatting rules** for different styles
4. **Add queue system** (Redis + Celery) for scalability
5. **Add monitoring** and health checks
6. **Implement rate limiting** for API endpoints

## 📝 Notes

- **Authentication**: Now uses real Supabase JWT verification
- **Database**: Fully integrated with your Supabase schema
- **File Storage**: Uses Supabase Storage with presigned URLs
- **Document Processing**: Only the formatting logic is mocked
- **Security**: Proper authentication and authorization implemented
- **Scalability**: Ready for production deployment

This backend provides a solid foundation for your document formatting service with real authentication and database integration.
