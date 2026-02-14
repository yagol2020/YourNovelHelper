import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_generate_endpoint_no_model(client):
    response = client.post("/generate", json={"prompt": "测试小说开头"})
    assert response.status_code == 503


def test_generate_endpoint_with_params(client):
    response = client.post(
        "/generate",
        json={
            "prompt": "测试小说开头",
            "style": "悬疑",
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
        },
    )
    assert response.status_code == 503


def test_generate_endpoint_validation(client):
    response = client.post("/generate", json={})
    assert response.status_code == 422
