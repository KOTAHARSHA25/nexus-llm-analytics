def test_health_check(client):
    """Verify that the health check endpoint returns 200 OK"""
    print(f"\nDEBUG: App Routes: {[r.path for r in client.app.routes]}")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data

def test_root_redirect(client):
    """Verify that root redirects to something valid or returns 200/404"""
    # Main might redirect to /docs or have a welcome message
    response = client.get("/")
    assert response.status_code in [200, 307]

def test_settings_retrieval(client):
    """Verify that settings can be retrieved"""
    response = client.get("/api/settings")
    # Note: Some deployments might require auth, but current backend is open
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "system" in data

def test_docs_page(client):
    """Verify that Swagger UI is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text
