import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Analyse de Prescription de Médicaments", layout="wide")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 36px;
        color: #1a5276;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }
    .sub-header {
        font-size: 24px;
        color: #2e4053;
        margin-top: 20px;
        margin-bottom: 10px;
        font-family: 'Arial', sans-serif;
    }
    .project-description {
        font-size: 16px;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Arial', sans-serif;
    }
    .member-card {
        border: 1px solid #dcdcdc;
        border-radius: 10px;
        padding: 20px;
        background-color: #f5f6fa;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: auto;
    }
    .member-card img {
        width: 20px;
        height: 20px;
        vertical-align: middle;
        margin-right: 8px;
    }
    .member-card p {
        margin: 10px 0;
        font-size: 16px;
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .member-card a {
        color: #2980b9;
        text-decoration: none;
    }
    .member-card a:hover {
        text-decoration: underline;
        color: #1a5276;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de l'état de la session
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'drug_labels' not in st.session_state:
    st.session_state.drug_labels = None

# Navigation dans la barre latérale
st.sidebar.title("Analyse de Prescription d'un médicament")
menu = st.sidebar.selectbox("Sélectionner le Menu", [
    "Accueil",
    "Chargement des Données",
    "Statistiques Descriptives",
    "Visualisations",
    "Entraînement du Modèle",
    "Prédiction pour Patient"
])

# Fonction d'encodage manuel pour les variables catégoriques
def manual_encoder(df, columns):
    df_encoded = df.copy()
    for var in columns:
        if var == 'BP':
            df_encoded[var] = df_encoded[var].replace(['LOW', 'NORMAL', 'HIGH'], [1, 2, 3])
        elif var == 'Cholesterol':
            df_encoded[var] = df_encoded[var].replace(['NORMAL', 'HIGH'], [1, 2])
        elif var == 'Drug':
            # Définir l'ordre des modalités pour Drug
            drug_labels = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
            drug_values = [0, 1, 2, 3, 4]  # Start at 0 for compatibility with RandomForestClassifier
            df_encoded[var] = df_encoded[var].replace(drug_labels, drug_values)
            # Stocker drug_labels pour le décodage
            if 'Drug' in columns and st.session_state.drug_labels is None:
                st.session_state.drug_labels = drug_labels
    return df_encoded

# Fonction de prétraitement des données
def preprocess_data(df):
    df_cleaned = df.copy()
    # Calculer le ratio Na/K
    df_cleaned['Na_sur_K'] = df_cleaned['Na'] / df_cleaned['K']
    # Supprimer la colonne 'Sex' car non corrélée avec 'Drug'
    df_cleaned = df_cleaned.drop('Sex', axis=1)
    # Encoder les variables catégoriques
    cat_columns = ['BP', 'Cholesterol']
    df_encoded = manual_encoder(df_cleaned, cat_columns)
    return df_encoded

# Section Accueil
if menu == "Accueil":
    st.markdown('<div class="main-header">Analyse de Prescription de Médicaments</div>', unsafe_allow_html=True)
    
    # Image d'en-tête
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014967.png", width=200, caption="Analyse des prescriptions médicales")
    
    # Description du projet
    st.markdown("""
        <div class="project-description">
            Bienvenue dans notre application d'analyse et de prédiction de prescriptions médicales. <br>
            Cette plateforme utilise un modèle Random Forest pour recommander le médicament le plus adapté à un patient en fonction de son âge, de sa pression artérielle, de son cholestérol, et des niveaux de sodium et potassium. Explorez les données, visualisez les tendances, et effectuez des prédictions précises grâce à une interface intuitive.
        </div>
    """, unsafe_allow_html=True)
    
    # Présentation du membre
    st.markdown('<div class="sub-header">Membre du projet</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div class="member-card">
                    <p><img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png" width="20"> <strong>Mamadou Lamarana Diallo</strong></p>
                    <p><img src="https://cdn-icons-png.flaticon.com/512/281/281769.png" width="20"> <a href="mailto:mamadoulamaranadiallomld1@gmail.com">mamadoulamaranadiallomld1@gmail.com</a></p>
                    <p><img src="https://cdn-icons-png.flaticon.com/512/724/724715.png" width="20"> <a href="tel:+221771050342">+221 771050342</a></p>
                    <p><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> <a href="https://www.linkedin.com/in/mamadou-lamarana-diallo-937430274/" target="_blank">LinkedIn</a></p>
                </div>
            """, unsafe_allow_html=True)

# Section de chargement des données
if menu == "Chargement des Données":
    st.title("Chargement des Données")
    uploaded_file = st.file_uploader("Importer votre fichier Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        if st.session_state.df is None or uploaded_file.name != st.session_state.last_uploaded_filename:
            try:
                df_temp
