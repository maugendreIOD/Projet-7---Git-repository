from flask import Flask, request, jsonify
import json
import joblib  # Utilisé pour charger un modèle scikit-learn
import numpy as np
import os
from sklearn.linear_model import LogisticRegression  # Pour le modèle de test
import pandas as pd

app = Flask(__name__)

# Test de test github actions test 5

# Détermine si nous sommes en mode test
is_testing = os.environ.get('FLASK_TESTING') == 'true'

# Charge le modèle ou crée un modèle fictif pour les tests
try:
    if is_testing:
        # Crée un modèle fictif pour les tests
        model = LogisticRegression()
        # Données factices pour l'entraînement
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)
    else:
        # Charge le vrai modèle en production
        model_path = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')
        model = joblib.load(model_path)
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
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

        # Utiliser le modèle pour prédire
        prediction = model.predict(features_array)
        # Ajout pour obtenir la probabilité
        probability = model.predict_proba(features_array)[0]
        
        # Retourner la prédiction et les explications SHAP sous forme de JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'probability': probability[1]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

