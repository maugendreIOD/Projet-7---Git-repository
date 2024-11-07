import pytest
import json
import os
from app.main import app
import pathlib

@pytest.fixture
def client():
    app.config['TESTING'] = True
    os.environ['FLASK_TESTING'] = 'true'
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_data():
    """Charge les données de test depuis le fichier JSON"""
    test_data_path = pathlib.Path(__file__).parent / 'test_data.json'
    with open(test_data_path) as f:
        return json.load(f)

def test_predict_endpoint_with_sample_data(client):
    """Test avec une seule feature simple - utile pour les tests de base"""
    simple_data = {
        "features": [2.5]
    }
    response = client.post('/predict', json=simple_data)
    assert response.status_code == 200
    assert "prediction" in response.json

def test_predict_endpoint_with_real_data(client, test_data):
    """Test avec les vraies données du fichier test_data.json"""
    # Accédez au premier élément de la liste (dictionnaire de caractéristiques)
    payload = {"features": list(test_data[0].values())}
    
    # Envoyer la requête à l'API
    response = client.post('/predict', json=payload)
    
    # Afficher un message d'erreur si la réponse n'est pas 200
    if response.status_code != 200:
        print("Erreur retournée par l'API :", response.json)  # Afficher l'erreur retournée par l'API
    assert response.status_code == 200
    assert "prediction" in response.json



def test_predict_endpoint_with_empty_features(client):
    """Test avec des features vides pour vérifier la gestion d'erreur"""
    empty_data = {
        "features": []
    }
    response = client.post('/predict', json=empty_data)
    assert response.status_code == 400
    assert "error" in response.json

def test_predict_endpoint_with_invalid_data(client):
    """Test avec des données invalides pour vérifier la gestion d'erreur"""
    invalid_data = {
        "wrong_key": [1, 2, 3]
    }
    response = client.post('/predict', json=invalid_data)
    assert response.status_code == 400
    assert "error" in response.json
