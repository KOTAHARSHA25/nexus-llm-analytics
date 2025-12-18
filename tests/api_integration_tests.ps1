# API Integration Test Script for Nexus-LLM-Analytics
# Run these commands in your terminal (PowerShell)

# 1. Test /upload-documents with CSV
curl -X POST "http://127.0.0.1:8000/upload-documents/" -F "file=@backend/data/StressLevelDataset.csv"

# 2. Test /upload-documents with PDF
curl -X POST "http://127.0.0.1:8000/upload-documents/" -F "file=@backend/data/HARSHA Kota Resume.pdf"

# 3. Test /upload-documents with TXT (create a sample if needed)
# echo "Sample text file for upload test." > backend/data/sample.txt
curl -X POST "http://127.0.0.1:8000/upload-documents/" -F "file=@backend/data/sample.txt"

# 4. Test /analyze endpoint with a valid query and file
curl -X POST "http://127.0.0.1:8000/analyze/" -H "Content-Type: application/json" -d '{"query": "summarize", "filename": "analyze.json"}'

# 5. Test /analyze endpoint with a RAG query (PDF)
curl -X POST "http://127.0.0.1:8000/analyze/" -H "Content-Type: application/json" -d '{"query": "rag", "filename": "HARSHA Kota Resume.pdf"}'

# 6. Test /generate-report endpoint (if implemented)
# curl -X POST "http://127.0.0.1:8000/generate-report/" -H "Content-Type: application/json" -d '{"filename": "analyze.json"}'

# 7. Test /upload-documents with invalid file type
curl -X POST "http://127.0.0.1:8000/upload-documents/" -F "file=@backend/data/1.json"

# 8. Test /upload-documents with large file (if available)
# curl -X POST "http://127.0.0.1:8000/upload-documents/" -F "file=@backend/data/largefile.csv"

# 9. Test /analyze with malformed query
curl -X POST "http://127.0.0.1:8000/analyze/" -H "Content-Type: application/json" -d '{"query": 123, "filename": "analyze.json"}'
