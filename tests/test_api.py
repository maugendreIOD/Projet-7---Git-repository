import pytest
import json
import os
from app.main import app
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

# Définir le pipeline de prétraitement
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

def test_predict_endpoint_with_sample_data(client):
    """Test avec un ensemble de 536 features factices pour correspondre au modèle"""
    # Générer 536 caractéristiques factices
    simple_data = {"features": [2.5] * 536}  # Liste de 536 valeurs identiques, ici 2.5 pour le test
    
    # Prétraitez les données factices via le pipeline (adapter le reshape si nécessaire)
    processed_data = preprocessing_pipeline.transform(np.array(simple_data["features"]).reshape(1, -1))

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
    
    # Prétraitez les données factices via le pipeline (adapter le reshape si nécessaire)
    processed_data = preprocessing_pipeline.transform(np.array(payload["features"]).reshape(1, -1))

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

