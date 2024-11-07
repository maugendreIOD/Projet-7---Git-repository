import pytest
import json
from app.main import app  # Importez directement app depuis main.py

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    # Charger les données depuis le fichier JSON
    with open('test_data.json') as f:
        data = json.load(f)

    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert "prediction" in response.json
