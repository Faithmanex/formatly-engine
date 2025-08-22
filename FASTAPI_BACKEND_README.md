# Formatly FastAPI Backend

This is a sample FastAPI backend for the Formatly document formatting application. It provides mock implementations of all the API endpoints that the frontend expects.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Run the server:**
   \`\`\`bash
   python main.py
   \`\`\`

3. **Access the API:**
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

4. **Set environment variables (optional):**
   - `PORT`: 8000 (automatically set by Render)

5. **Deploy and get your service URL**

## 📋 API Endpoints

### Core Document Processing
- `POST /api/documents/upload` - Upload documents
- `POST /api/documents/process` - Start document formatting
- `GET /api/documents/status/{job_id}` - Check processing status
- `GET /api/documents/download/{job_id}` - Download formatted document

### Configuration
- `GET /api/formatting/styles` - Get available formatting styles
- `GET /api/formatting/variants` - Get English variants

### Testing Utilities
- `GET /api/jobs` - List all processing jobs
- `DELETE /api/jobs/{job_id}` - Delete a job
- `GET /api/files` - List uploaded files

## 🔧 Configuration

### Frontend Integration

Update your frontend's environment variables:

\`\`\`env
FASTAPI_BASE_URL=https://your-render-service.onrender.com
FASTAPI_TIMEOUT=30000
\`\`\`

### CORS Configuration

The backend is configured to allow all origins for testing. For production, update the CORS settings in `main.py`:

\`\`\`python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
\`\`\`

## 🧪 Testing

### Health Check
\`\`\`bash
curl https://your-service.onrender.com/health
\`\`\`

### Upload Test
\`\`\`bash
curl -X POST "https://your-service.onrender.com/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-document.txt"
\`\`\`

### Process Document Test
\`\`\`bash
curl -X POST "https://your-service.onrender.com/api/documents/process" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.txt",
    "content": "VGVzdCBkb2N1bWVudCBjb250ZW50",
    "style": "apa",
    "englishVariant": "us",
    "reportOnly": false,
    "includeComments": true,
    "preserveFormatting": true
  }'
\`\`\`

## 🔄 Mock Behavior

This sample backend simulates real document processing:

1. **Upload**: Stores files in memory with unique IDs
2. **Processing**: Creates jobs with simulated 4-second processing time
3. **Status Updates**: Progress from "queued" → "processing" → "formatted"
4. **Download**: Returns mock formatted content with metadata

## 🚀 Production Considerations

For a production backend, you would need to:

1. **Replace in-memory storage** with Redis/database
2. **Implement real JWT verification** with Supabase
3. **Add actual document processing** logic
4. **Implement proper error handling** and logging
5. **Add rate limiting** and security measures
6. **Use environment variables** for configuration
7. **Add monitoring** and health checks

## 📝 Notes

- This is a **sample/mock implementation** for testing purposes
- All document processing is simulated
- Data is stored in memory and will be lost on restart
- No authentication is enforced (returns mock user data)
- CORS is configured for development (allows all origins)

Use this backend to test your frontend integration before implementing the actual document processing logic.
