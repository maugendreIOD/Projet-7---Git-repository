import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image
import joblib
import os




# URL de ton API Flask locale pour la prédiction
api_url = 'https://openclassrooms-api-2-262775559739.us-central1.run.app/predict'

explainer_path = os.path.join("streamlit", "shap_explainer.joblib")
explainer = joblib.load(explainer_path)

pipeline_retenu_path = os.path.join("streamlit", "preprocessing_pipeline.pkl")
pipeline_retenu = joblib.load(pipeline_retenu_path)

top_features_path = os.path.join("streamlit", 'top_features.csv')

# Configurer la page Streamlit
st.set_page_config(page_title="Prédictions d'un modèle de scoring bancaire", page_icon="dart", layout="wide", initial_sidebar_state="auto")

# Charger et afficher le logo
logo_path = os.path.join("streamlit", "logo_open_classroom.png")
logo = Image.open(logo_path)

st.image(logo, width=700)
st.title("Prédictions d'un modèle de scoring bancaire")

# Charger le fichier d'importance des features
try:
    top_features = pd.read_csv(top_features_path)
except FileNotFoundError:
    st.error("Le fichier 'top_features.csv' est introuvable.")

st.sidebar.header("Ajouter les données")
# Uploader le fichier:
uploaded_file = st.sidebar.file_uploader("",type="json", key="fileUploader")

st.sidebar.header("Analyses")
page_options = ["Analyse dossier d'emprunt", "Analyse données globales"]
# Use st.radio to allow selecting only one option
selected_page = st.sidebar.radio("",page_options)

# Initialiser la variable dans session_state si elle n'existe pas
if 'selected_data' not in st.session_state:
    st.session_state['selected_data'] = None

if 'data' not in st.session_state:
    st.session_state['data'] = None


if uploaded_file is not None:
    try:
        # Charger le fichier JSON dans un DataFrame
        data = pd.read_json(uploaded_file)
        st.session_state['data'] = data

        # Vérifier que la colonne 'SK_ID_CURR' existe
        if 'SK_ID_CURR' not in data.columns:
            st.error("Le fichier JSON doit contenir une colonne 'SK_ID_CURR'.")
        else:
            if selected_page == "Analyse dossier d'emprunt":
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
                        st.session_state['selected_data'] = data[data['SK_ID_CURR'] == selected_sk_id].drop(columns=['SK_ID_CURR'])
                        
                        # Afficher les données de l'individu sélectionné
                        with st.expander("Données de l'individu sélectionné :"):
                            st.write(selected_data)
                        
                        # Transformez les données sélectionnées en liste de valeurs plutôt qu'en dictionnaire
                        payload = {"features": selected_data.iloc[0].values.tolist()}

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
                                color = "#FF0000"  # Couleur rouge pour rejeté
                            
                            # Création de la figure
                            fig = go.Figure()
                            
                            # Ajouter le trace Indicator
                            fig.add_trace(go.Indicator(
                                mode="gauge+number+delta",
                                value=probability_reject,
                                title={'text': "Probabilité de rejet (%)"},
                                delta={
                                    'reference': 50,
                                    'increasing': {'color': "red"},   # Couleur rouge pour une augmentation au-dessus de la référence
                                    'decreasing': {'color': "green"} # Couleur verte pour une baisse en dessous de la référence
                                },
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "red" if probability_reject > 50 else "green"},
                                    'steps': [
                                        {'range': [0, 50], 'color': 'lightgreen'},
                                        {'range': [50, 100], 'color': 'pink'}
                                    ]
                                }
                            ))

                            # Ajouter des annotations (libellés explicatifs)
                            fig.add_annotation(
                                x=0.2, y=0.5,
                                text="Risque faible",
                                showarrow=False,
                                font=dict(size=12, color="darkgreen"),
                                align="center"
                            )

                            fig.add_annotation(
                                x=0.8, y=0.5,
                                text="Risque élevé",
                                showarrow=False,
                                font=dict(size=12, color="darkred"),
                                align="center"
                            )

                            # Rendre le graphique responsive
                            fig.update_layout(
                                annotations=[
                                    dict(
                                        x=0.2, y=0.5,
                                        text="Risque faible",
                                        showarrow=False,
                                        font=dict(size=12, color="darkgreen"),
                                        align="center"
                                    ),
                                    dict(
                                        x=0.8, y=0.5,
                                        text="Risque élevé",
                                        showarrow=False,
                                        font=dict(size=12, color="darkred"),
                                        align="center"
                                    )
                                ]
                            )
                            st.plotly_chart(fig)
                            st.write("Graphique de jauge indiquant la probabilité de rejet (%) basée sur les caractéristiques du client.")

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


                        else:
                            st.error(f"Erreur dans la réponse de l'API : {response.json().get('error', 'Erreur inconnue')}")

                        

                    
                # Colonne droite : fiche client
                with col2:
                    st.header("Fiche client")
                    if st.session_state['selected_data'] is not None:
                        selected_data = st.session_state['selected_data']

                        # Convertir 0/1 en Homme/Femme
                        genre = "Homme" if selected_data['CODE_GENDER'].iloc[0] == 1 else "Femme"
                        age = int(selected_data['DAYS_BIRTH'].iloc[0]/(-365))
                        # Stylisation de la fiche client
                        st.markdown(f"""
                            <div style="
                                background-color: #f9f9f9;
                                padding: 20px;
                                border-radius: 10px;
                                border: 1px solid #ddd;
                                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                                font-family: Arial, sans-serif;
                                color: #333;
                                ">
                                <h3 style="text-align: center; color: #001F54;">Client - {selected_sk_id}</h3>
                                <ul style="list-style: none; padding: 0; font-size: 16px;">
                                    <li><strong>Genre :</strong> {genre}</li>
                                    <li><strong>Âge :</strong> {age} ans</li>
                                    <li><strong>Revenu annuel :</strong> {round(selected_data['AMT_INCOME_TOTAL'].iloc[0]):,} €</li>
                                    <li><strong>Montant emprunt :</strong> {round(selected_data['AMT_CREDIT'].iloc[0]):,} €</li>
                                    <li><strong>Remboursement annuel :</strong> {round(selected_data['AMT_ANNUITY'].iloc[0]):,} €</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                
                col1A, col2A = st.columns([1, 1])

                with col1A:
                    st.header("Variables majeures dans la prédiction")
                    if st.session_state['selected_data'] is not None:
                        selected_data = st.session_state['selected_data']
                        selected_data_processed = pipeline_retenu.transform(selected_data)

                        # Utiliser la prédiction pour calculer les valeurs SHAP localement
                        # Calcul des valeurs SHAP localement
                        shap_values = explainer.shap_values(selected_data_processed)

                        # Waterfall Plot
                        st.subheader("Waterfall Plot (SHAP)")
                        explanation = shap.Explanation(
                            values=shap_values[0],
                            base_values=explainer.expected_value,
                            data=selected_data_processed[0],
                            feature_names=selected_data.columns
                        )
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        st.pyplot(fig)
                        st.write("Le graphique Waterfall SHAP ci-dessous explique les contributions des caractéristiques individuelles au résultat prédictif.")

                with col2A:
                    st.header("Top 10 des variables importantes du modèle")

                    if st.session_state['selected_data'] is not None:
                        selected_data = st.session_state['selected_data']
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

            if selected_page == "Analyse données globales":
                    
                    if st.session_state['selected_data'] is not None and st.session_state['data'] is not None:
                        selected_data = st.session_state['selected_data']
                        data = st.session_state['data']

                        col3, col4 = st.columns([1, 1])

                        with col3:

                            st.header("Analyse des caractéristiques du client par rapport au panel")
                            # Liste des caractéristiques disponibles
                            features = selected_data.columns.tolist()

                            # Sélecteur interactif pour choisir une caractéristique
                            selected_feature = st.selectbox("Choisissez une caractéristique à visualiser :", features)
                            
                            # Valeur pour le client sélectionné
                            selected_value = selected_data[selected_feature].iloc[0]

                            # Histogramme stylisé
                            plt.figure(figsize=(10, 6))
                            plt.hist(data[selected_feature], bins=30, alpha=0.6, color='#4682B4', edgecolor='white', linewidth=1.5, label='Tous les clients')

                            # Ligne verticale pour le client sélectionné
                            plt.axvline(selected_value, color='#FF4500', linestyle='--', linewidth=3, label='Client sélectionné')

                            # Ajouter un fond légèrement gris
                            plt.gca().set_facecolor('#f9f9f9')

                            # Titre et labels personnalisés
                            plt.title(f'Distribution de {selected_feature}', fontsize=16, fontweight='bold', color='#333333')
                            plt.xlabel(selected_feature, fontsize=14, color='#555555')
                            plt.ylabel('Fréquence', fontsize=14, color='#555555')

                            # Style des ticks
                            plt.xticks(fontsize=12, color='#444444')
                            plt.yticks(fontsize=12, color='#444444')

                            # Légende stylisée
                            plt.legend(fontsize=12, frameon=True, shadow=True, loc='upper right')

                            # Bordure autour du graphique
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['left'].set_color('#bbbbbb')
                            plt.gca().spines['bottom'].set_color('#bbbbbb')

                            # Afficher le graphique dans Streamlit
                            st.pyplot(plt)

                            st.markdown(f"""
                            ### Analyse de {selected_feature}
                            - La ligne rouge représente la valeur du client sélectionné.
                            - La distribution montre la répartition de {selected_feature} parmi les autres clients.
                            - Les chargés de relation peuvent ainsi voir si ce client est "typique" ou se distingue des autres clients sur cette caractéristique.
                            """)
                        
                        with col4:
                            st.header("Analyse croisée des variables du panel")
                            # Widgets pour sélectionner deux variables
                            feature_x = st.selectbox("Choisissez la première caractéristique (X-axis) :", data.drop(columns='SK_ID_CURR').columns.tolist())
                            feature_y = st.selectbox("Choisissez la deuxième caractéristique (Y-axis) :", data.drop(columns='SK_ID_CURR').columns.tolist())

                            plt.figure(figsize=(10, 6))

                            # Création du scatter plot
                            plt.scatter(data[feature_x], data[feature_y], alpha=0.6, color='#4682B4', edgecolors='white', linewidth=0.7)

                            # Titre et étiquettes
                            plt.title(f'Relation entre {feature_x} et {feature_y}', fontsize=16, fontweight='bold', color='#333333')
                            plt.xlabel(feature_x, fontsize=14, color='#555555')
                            plt.ylabel(feature_y, fontsize=14, color='#555555')

                            # Grille stylisée
                            plt.grid(color='#dddddd', linestyle='--', linewidth=0.7, alpha=0.7)

                            st.pyplot(plt)

                            # Valeurs pour le client sélectionné
                            x_value = selected_data[feature_x].iloc[0]
                            y_value = selected_data[feature_y].iloc[0]

                            plt.scatter(x_value, y_value, color='#FF4500', s=100, label='Client sélectionné', edgecolors='black', linewidth=1.5)

                            # Annotation pour le client sélectionné
                            plt.annotate("Client sélectionné", (x_value, y_value), textcoords="offset points", xytext=(10, 10), ha='center',
                                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

                            # Légende
                            plt.legend(fontsize=12, frameon=True, loc='upper right')

                            # Identifier si l'une des variables est catégorique
                            is_feature_x_categorical = data[feature_x].nunique() < 10
                            is_feature_y_categorical = data[feature_y].nunique() < 10

                            if is_feature_x_categorical or is_feature_y_categorical:
                                # Si l'une des variables est catégorique, afficher un boxplot
                                plt.figure(figsize=(10, 6))
                                if is_feature_x_categorical:
                                    sns.boxplot(x=feature_x, y=feature_y, data=data, palette="viridis")
                                else:
                                    sns.boxplot(x=feature_y, y=feature_x, data=data, palette="viridis")

                                # Ajouter le point du client sélectionné
                                plt.scatter(x=x_value, y=y_value, color='#FF4500', s=100, label='Client sélectionné', edgecolors='black', linewidth=1.5)
                                plt.legend(fontsize=12)
                                plt.title(f"Boxplot de {feature_x} et {feature_y}")
                                st.pyplot(plt)

                            else:
                                # Si les deux variables sont continues, afficher un scatter plot
                                plt.figure(figsize=(10, 6))
                                plt.scatter(data[feature_x], data[feature_y], alpha=0.6, color='#4682B4', edgecolors='white', linewidth=0.7, label='Tous les clients')

                                # Ajouter la valeur du client sélectionné
                                plt.scatter(x_value, y_value, color='#FF4500', s=100, label='Client sélectionné', edgecolors='black', linewidth=1.5)

                                plt.title(f'Relation entre {feature_x} et {feature_y}', fontsize=16, fontweight='bold', color='#333333')
                                plt.xlabel(feature_x, fontsize=14, color='#555555')
                                plt.ylabel(feature_y, fontsize=14, color='#555555')
                                plt.legend(fontsize=12)
                                st.pyplot(plt)

                            # Ajouter une heatmap seulement si les deux variables sont continues et calcul de corrélation possible
                            if not is_feature_x_categorical and not is_feature_y_categorical:
                                heatmap_data = data[[feature_x, feature_y]].corr()

                                plt.figure(figsize=(8, 6))
                                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", cbar=True, square=True)
                                plt.title(f"Heatmap de {feature_x} et {feature_y}")
                                st.pyplot(plt)

                            # Explication interactive
                            st.markdown(f"""
                            ### Analyse Bi-variée
                            - Ce graphique montre la relation entre **{feature_x}** et **{feature_y}**.
                            - Le point rouge représente la position du client sélectionné.
                            - Vous pouvez identifier si ce client est une exception ou suit les tendances générales.
                            """)

                    else:
                        st.warning("Aucun client sélectionné. Veuillez d'abord choisir un client dans l'onglet 'Analyse dossier d'emprunt'.")




    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier JSON : {str(e)}")
