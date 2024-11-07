import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# URL de ton API Flask locale
api_url = 'https://openclassrooms-api-262775559739.us-central1.run.app/predict'

# Configurer la page Streamlit
st.set_page_config(page_title="Prédictions d'un modèle de scoring bancaire", page_icon="dart", initial_sidebar_state="auto")

logo=Image.open("logo_open_classroom.png")
st.image(logo, width=700)
st.title("Prédictions d'un modèle de scoring bancaire")

# Chargement du fichier JSON
uploaded_file = st.file_uploader("Charger un fichier JSON contenant les features", type="json")

if uploaded_file is not None:
    try:
        # Charger le fichier JSON dans un dictionnaire Python
        data = json.load(uploaded_file)
        
        # Convertir le JSON en DataFrame pour faciliter la manipulation (optionnel, si nécessaire)
        df = pd.DataFrame(data)
        
        # Vérifier que la colonne 'SK_ID_CURR' existe
        if 'SK_ID_CURR' not in df.columns:
            st.error("Le fichier JSON doit contenir une colonne 'SK_ID_CURR'.")
        else:
            # Afficher les données chargées
            st.write("Données de features :")
            st.dataframe(df)
            
            # Ajouter une option par défaut dans le sélecteur
            sk_id_options = ["Sélectionner un ID"] + list(df['SK_ID_CURR'].unique())
            selected_sk_id = st.selectbox("Choisir un SK_ID_CURR pour prédire", sk_id_options)
            
             # Vérifier si un ID a été sélectionné
            if selected_sk_id != "Sélectionner un ID":
                # Filtrer le DataFrame pour l'individu sélectionné
                selected_data = df[df['SK_ID_CURR'] == selected_sk_id].drop(columns=['SK_ID_CURR']).to_dict(orient='records')[0]

                # Afficher les données de l'individu sélectionné dans une liste déroulante
                with st.expander("Données de l'individu sélectionné :"):
                    st.json(selected_data)
                
                # Préparer les données pour l'API
                payload = {"features": list(selected_data.values())}
                
                # Envoyer les données à l'API pour obtenir la prédiction
                response = requests.post(api_url, json=payload)
                
                # Si la réponse est positive, afficher la prédiction
                if response.status_code == 200:
                    prediction = response.json().get('prediction', [])
                    st.success(f'Prédiction pour SK_ID_CURR {selected_sk_id} : {prediction}')
                else:
                    st.error(f"Erreur dans la réponse de l'API : {response.json().get('error', 'Erreur inconnue')}")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier JSON : {str(e)}")