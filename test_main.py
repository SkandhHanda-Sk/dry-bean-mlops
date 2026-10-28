
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Ready to Serve"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    data = {
        "Area": 28395,
        "Perimeter": 610.291,
        "MajorAxisLength": 208.178117,
        "MinorAxisLength": 173.888747,
        "AspectRation": 1.197191,
        "Eccentricity": 0.549812,
        "ConvexArea": 28715,
        "EquivDiameter": 190.141097,
        "Extent": 0.763923,
        "Solidity": 0.988856,
        "roundness": 0.958027,
        "Compactness": 0.913358,
        "ShapeFactor1": 0.007332,
        "ShapeFactor2": 0.003147,
        "ShapeFactor3": 0.834222,
        "ShapeFactor4": 0.998724
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], str)


def test_predict_invalid_input():
    response = client.post("/predict", json={"Area": -1})
    assert response.status_code == 422
