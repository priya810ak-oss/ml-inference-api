from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["predict_endpoint"] == "POST /predict"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in ["setosa", "versicolor", "virginica"]
    assert set(data["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01

def test_predict_rejects_wrong_number_of_features():
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 422
