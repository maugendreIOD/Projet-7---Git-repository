from flask import Flask, request, jsonify
import json
import joblib  # Utilisé pour charger un modèle scikit-learn
import numpy as np
import os

app = Flask(__name__)

# Test de test github actions test 5
# Charger le modèle au démarrage de l'application
# Remplace 'model.pkl' par le chemin de ton fichier de modèle
model = joblib.load('linear_regression_model.pkl')

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

