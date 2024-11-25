import pytest
import json
import os
from app.main import app
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from app.main import app, preprocessing_pipeline


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
    """Test avec un ensemble de 536 features factices pour correspondre au modèle"""
    # Générer 536 caractéristiques factices
    simple_data = {"features": [2.5] * 536}  # Liste de 536 valeurs identiques, ici 2.5 pour le test
    
    # Convertir les features en tableau numpy (reshape nécessaire pour transformer 1 ligne)
    features_array = np.array(simple_data["features"]).reshape(1, -1)

    # Prétraitez les données factices via le pipeline (adapter le reshape si nécessaire)
    processed_data = preprocessing_pipeline.transform(features_array)

    # Envoyer la requête à l'API
    response = client.post('/predict', json={"features": processed_data.tolist()[0]})

    # Afficher un message d'erreur si la réponse n'est pas 200
    if response.status_code != 200:
        print("Erreur retournée par l'API :", response.json)  # Afficher l'erreur retournée par l'API
    
    assert response.status_code == 200
    assert "prediction" in response.json


def test_predict_endpoint_with_real_data(client, test_data):
    """Test avec les vraies données du fichier test_data.json"""
    # Créez le payload en extrayant les valeurs du dictionnaire
    payload = {"features": list(test_data.values())}
    
    # Convertir les features en tableau numpy (reshape nécessaire pour transformer 1 ligne)
    features_array = np.array(payload["features"]).reshape(1, -1)
    
    # Prétraitez les données factices via le pipeline (adapter le reshape si nécessaire)
    processed_data = preprocessing_pipeline.transform(features_array)

    # Envoyer la requête à l'API
    response = client.post('/predict', json={"features": processed_data.tolist()[0]})
    
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

