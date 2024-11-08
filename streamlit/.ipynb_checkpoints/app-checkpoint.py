import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image
import joblib

# URL de ton API Flask locale pour la prédiction
api_url = 'https://openclassrooms-api-262775559739.us-central1.run.app/predict'

# Charger l'explainer SHAP local
explainer = joblib.load('shap_explainer.joblib')  # Explainer SHAP local

# Configurer la page Streamlit
st.set_page_config(page_title="Prédictions d'un modèle de scoring bancaire", page_icon="dart", layout="wide", initial_sidebar_state="auto")

# Charger et afficher le logo
logo = Image.open("logo_open_classroom.png")
st.image(logo, width=700)
st.title("Prédictions d'un modèle de scoring bancaire")

# Charger le fichier d'importance des features
try:
    top_features = pd.read_csv('top_features.csv')
except FileNotFoundError:
    st.error("Le fichier 'top_features.csv' est introuvable.")
    
# Encapsuler la barre de téléchargement de fichier dans une colonne
col_upload, col_empty = st.columns([1, 4])  # Créer une colonne étroite pour la barre de chargement

with col_upload:
    # Charger le fichier JSON contenant les features
    uploaded_file = st.file_uploader("Charger un fichier JSON contenant les features", type="json")

if uploaded_file is not None:
    try:
        # Charger le fichier JSON dans un DataFrame
        data = pd.read_json(uploaded_file)
        
        # Vérifier que la colonne 'SK_ID_CURR' existe
        if 'SK_ID_CURR' not in data.columns:
            st.error("Le fichier JSON doit contenir une colonne 'SK_ID_CURR'.")
        else:
            # Interface en deux colonnes pour afficher les données et les explications
            col1, col2 = st.columns([1, 1])

            with col1:
               
                st.header("Analyse de dossier d'emprunt")
                
                # Ajouter une option par défaut dans le sélecteur
                sk_id_options = ["Sélectionner un ID"] + list(data['SK_ID_CURR'].unique())
                selected_sk_id = st.selectbox("Choisir un SK_ID_CURR pour prédire", sk_id_options)
                
                # Vérifier si un ID a été sélectionné
                if selected_sk_id != "Sélectionner un ID":
                    # Filtrer le DataFrame pour l'individu sélectionné
                    selected_data = data[data['SK_ID_CURR'] == selected_sk_id].drop(columns=['SK_ID_CURR'])
    
                    # Afficher les données de l'individu sélectionné
                    with st.expander("Données de l'individu sélectionné :"):
                        st.write(selected_data)
                    
                    # Préparer les données pour l'API (envoyer uniquement les features sans 'SK_ID_CURR')
                    payload = {"features": selected_data.values.tolist()[0]}  # Convertir les données en liste pour l'API
                    
                    # Envoyer les données à l'API pour obtenir la prédiction
                    response = requests.post(api_url, json=payload)
                        
                    # Si la réponse est positive, afficher la prédiction
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get('prediction', [])
                        probability_reject = result.get('probability', 0) * 100  # Probabilité de rejet en pourcentage
                        
                        # Afficher la prédiction dans Streamlit
                        st.success(f'Prédiction pour SK_ID_CURR {selected_sk_id} réalisée')

                        # Afficher le message de décision avec un encadré stylisé
                        if prediction[0] == 0:
                            decision = "Dossier d'emprunt accordé"
                            color = "#4CAF50"  # Couleur verte pour approuvé
                        else:
                            decision = "Dossier d'emprunt risqué"
                            color = "#FF5733"  # Couleur rouge pour rejeté
                        
                        # Encadré stylisé pour la décision et la probabilité
                        st.markdown(
                            f"""
                            <div style="
                                padding: 10px; 
                                border-radius: 5px; 
                                background-color: {color}; 
                                color: white; 
                                text-align: center;
                                font-size: 18px;
                                font-weight: bold;">
                                {decision}<br>
                                Probabilité de rejet: {probability_reject:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        # Utiliser la prédiction pour calculer les valeurs SHAP localement
                        selected_data_np = selected_data.values  # Convertir en NumPy array pour SHAP

                        # Calcul des valeurs SHAP localement
                        shap_values = explainer.shap_values(selected_data_np)

                        st.header("Variables importantes pour le dossier")
    
                        # Waterfall Plot
                        st.subheader("Waterfall Plot (SHAP)")
                        explanation = shap.Explanation(
                            values=shap_values[0],
                            base_values=explainer.expected_value,
                            data=selected_data_np[0],
                            feature_names=selected_data.columns
                        )
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(explanation, max_display=10)
                        st.pyplot(fig)
                    else:
                        st.error(f"Erreur dans la réponse de l'API : {response.json().get('error', 'Erreur inconnue')}")

                    

                
            # Colonne droite : visualisations SHAP
            with col2:
                st.header("Top 10 des variables importantes du modèle")

                # Création de l'histogramme stylisé
                plt.figure(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
                bars = plt.barh(top_features['nom_colonne'], top_features['importance'], color=colors, edgecolor='black', linewidth=1.5)
        
                for bar in bars:
                    bar.set_linewidth(1.2)
                    bar.set_edgecolor('black')
                    bar.set_alpha(0.85)
        
                for i, (importance, feature) in enumerate(zip(top_features['importance'], top_features['nom_colonne'])):
                    plt.text(importance + 0.01, i, f'{importance:.2f}', va='center', fontsize=10, fontweight='bold', color='black')
        
                plt.xlabel('Importance', fontsize=12)
                plt.title('Top 10 Variables Importantes', fontsize=14)
                plt.gca().invert_yaxis()
                
                # Afficher le graphique dans Streamlit
                st.pyplot(plt)
                
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier JSON : {str(e)}")
