from fastapi.testclient import TestClient
from ...api.main import app

client = TestClient(app)

def test_read_koi_data():
    response = client.get("/koi/")
    assert response.status_code == 200
    assert response.json() == {"message": "KOI data will be here"}

