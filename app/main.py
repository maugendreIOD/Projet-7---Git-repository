from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import json
import joblib  # Utilisé pour charger un modèle scikit-learn
import numpy as np
import os
from sklearn.linear_model import LogisticRegression  # Pour le modèle de test
import pandas as pd

app = Flask(__name__)

# Test de test github actions test 22

# Définir le pipeline de prétraitement
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

# Détermine si nous sommes en mode test
is_testing = os.environ.get('FLASK_TESTING') == 'true'

try:
    if is_testing:
        # Crée un modèle fictif pour les tests
        model = LogisticRegression()
        X = np.random.rand(10, 536)  # Exemple de données (10 exemples, 536 caractéristiques)
        y = np.random.randint(0, 2, 10)
        
        # Ajuste le pipeline sur les données brutes
        preprocessing_pipeline.fit(X)
        
        # Prétraite les données pour l'entraînement du modèle
        X_processed = preprocessing_pipeline.transform(X)
        
        # Entraîne le modèle sur les données prétraitées
        model.fit(X_processed, y)
    else:
        # Charge le vrai modèle et le pipeline de prétraitement en production
        model_path = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')
        pipeline_path = os.path.join(os.path.dirname(__file__), 'preprocessing_pipeline.pkl')
        model = joblib.load(model_path)
        preprocessing_pipeline = joblib.load(pipeline_path)
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du pipeline : {str(e)}")
    raise


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du corps de la requête
        data = request.get_json()
        features = data.get('features', [])

        # Vérifier que les features sont non vides
        if not features:
            return jsonify({'error': 'Aucune feature fournie'}), 400

        # Convertir les features en un format compatible avec ton modèle (ex: numpy array)
        features_array = np.array(features).reshape(1, -1)  # Reshape si une seule instance

        # Appliquer le pipeline de prétraitement
        processed_features = preprocessing_pipeline.transform(features_array)

        # Utiliser le modèle pour prédire
        prediction = model.predict(processed_features)
        # Ajout pour obtenir la probabilité
        probability = model.predict_proba(processed_features)[0]
        
        # Retourner la prédiction et les explications SHAP sous forme de JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'probability': probability[1]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

