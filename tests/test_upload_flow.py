def test_upload_csv(client, sample_csv_file):
    """Test uploading a valid CSV file"""
    with open(sample_csv_file, "rb") as f:
        files = {"file": ("test_data.csv", f, "text/csv")}
        response = client.post("/api/upload/", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_data.csv"
    assert "file_id" in data
    assert data["status"] == "success"

def test_upload_no_file(client):
    """Test upload without a file attached"""
    response = client.post("/api/upload/")
    # FastAPI returns 422 for missing required fields
    assert response.status_code == 422

def test_upload_bad_extension(client, tmp_path):
    """Test uploading a forbidden file extension"""
    d = tmp_path / "bad"
    d.mkdir()
    p = d / "malware.exe"
    p.write_text("fake malware")
    
    with open(p, "rb") as f:
        files = {"file": ("malware.exe", f, "application/x-msdownload")}
        response = client.post("/api/upload/", files=files)
    
    # Should be rejected by security validation
    assert response.status_code in [400, 422]
    if response.status_code == 400:
        assert "not allowed" in response.json().get("detail", "").lower()

def test_upload_txt_rag(client, sample_text_file):
    """Test uploading a text file for RAG"""
    with open(sample_text_file, "rb") as f:
        files = {"file": ("test_doc.txt", f, "text/plain")}
        response = client.post("/api/upload/", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "test_doc.txt" in data["filename"]
