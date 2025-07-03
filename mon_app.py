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
st.sidebar.title("Analyse des Prescriptions Médicales")
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
    st.markdown("## 🩺 **Analyse de Prescription d'un médicament**")
    st.markdown("---")
    
    # Description du projet
    st.markdown("""🎯 **Ce projet exploite les données médicales des patients pour recommander le traitement le plus adapté à leur profil, 
    en s’appuyant sur des techniques de machine learning et d’analyse prédictive.**""")
    
    # Espacement vertical
    st.markdown("---")
    st.markdown("## 👤 Informations personnelles")
    st.markdown("---")
    
    # Organisation des infos de contact
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**👤 Nom**")
        st.markdown("**Mamadou Lamarana Diallo**")
    
    with col2:
        st.markdown("**📧 Email**")
        st.markdown("[mamadoulamaranadiallomld1@gmail.com](mailto:mamadoulamaranadiallomld1@gmail.com)")
    
    with col3:
        st.markdown("**📞 Contact**")
        st.markdown("[+221 771050342](tel:+221771050342)")
    
    with col4:
        st.markdown("**🔗 LinkedIn**")
        st.markdown("""<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align: middle;"> [LinkedIn](https://www.linkedin.com/in/mamadou-lamarana-diallo-937430274/)""", unsafe_allow_html=True)

# Section de chargement des données
if menu == "Chargement des Données":
    st.title("Chargement des Données")
    uploaded_file = st.file_uploader("Importer votre fichier Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        if st.session_state.df is None or uploaded_file.name != st.session_state.last_uploaded_filename:
            try:
                df_temp = pd.read_excel(uploaded_file)
                st.session_state.df_raw = df_temp.copy()  # Conserver les données brutes
                st.session_state.df = preprocess_data(df_temp)
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.success("Fichier chargé et traité avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")
                st.session_state.df = None
                st.session_state.df_raw = None

# Section des statistiques descriptives
if menu == "Statistiques Descriptives":
    st.title("Statistiques Descriptives")
    if st.session_state.df is not None and st.session_state.df_raw is not None:
        df = st.session_state.df
        df_raw = st.session_state.df_raw
        
        # Aperçu des données brutes
        st.subheader("Aperçu des Données Brutes")
        st.dataframe(df_raw.head(10))
        
        # Informations de base
        st.subheader("Informations de Base")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Nombre de lignes : {df.shape[0]}")
            st.write(f"Nombre de colonnes : {df.shape[1]}")
        with col2:
            st.write("**Valeurs manquantes :**")
            # Utiliser df_raw pour inclure Sex
            missing_values = df_raw.isna().sum()
            for col, count in missing_values.items():
                st.write(f"{col}: {count}")
            st.write(f"**Vérification de données dupliquées :** {df.duplicated().sum()}")
        
        # Statistiques des variables numériques
        st.subheader("Statistiques des Variables Numériques")
        num_cols = ['Age', 'Na', 'K']
        st.dataframe(df[num_cols].describe().T)
        
        # Distribution des variables catégoriques
        st.subheader("Distribution des Variables Catégoriques")
        cat_cols = ['BP', 'Cholesterol', 'Drug']
        for col in cat_cols:
            st.write(f"\nDistribution de {col} :")
            # Utiliser df_raw pour afficher les modalités d'origine
            st.dataframe(df_raw[col].value_counts())
        
        # Analyse des valeurs aberrantes
        st.subheader("Analyse des Valeurs Aberrantes")
        for var in num_cols:
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.boxplot(data=df, x=var, ax=ax)
            ax.set_title(f"Boxplot de {var}")
            st.pyplot(fig)
    else:
        st.warning("Veuillez charger un fichier de données d'abord.")

# Section des visualisations
if menu == "Visualisations":
    st.title("Visualisations des Données")
    if st.session_state.df is not None and st.session_state.df_raw is not None:
        df = st.session_state.df
        df_raw = st.session_state.df_raw
        
        # Sous-menu pour univariée ou bivariée
        analysis_type = st.radio("Type d'Analyse", ["Analyse Univariée", "Analyse Bivariée"])
        
        if analysis_type == "Analyse Univariée":
            vis_option = st.selectbox("Sélectionner la Visualisation", [
                "Distribution des Variables Numériques",
                "Pairplot des Variables Numériques",
                "Distribution des Variables Qualitatives"
            ])
            
            # Distribution des variables numériques avec histplot
            if vis_option == "Distribution des Variables Numériques":
                num_cols = ['Age', 'Na', 'K']
                for var in num_cols:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.histplot(data=df, x=var, stat="density", label="Histogramme", ax=ax)
                    ax.set_xlabel(var)
                    ax.set_ylabel("Densité")
                    ax.set_title(f"Distribution de la variable {var}")
                    ax.legend()
                    st.pyplot(fig)
            
            # Pairplot des variables numériques
            elif vis_option == "Pairplot des Variables Numériques":
                fig = sns.pairplot(df_raw[['Age', 'Na', 'K', 'Drug']], hue='Drug', diag_kind='hist')
                plt.suptitle("Pairplot des Variables Numériques", y=1.02)
                st.pyplot(fig)
            
            # Distribution des variables qualitatives
            elif vis_option == "Distribution des Variables Qualitatives":
                cat_cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
                for var in cat_cols:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    df_raw[var].value_counts().plot.bar(ax=ax)
                    ax.set_xlabel(var)
                    ax.set_ylabel("Nombre")
                    ax.set_title(f"Distribution de {var}")
                    st.pyplot(fig)
        
        elif analysis_type == "Analyse Bivariée":
            vis_option = st.selectbox("Sélectionner la Visualisation", [
                "Âge vs Médicament",
                "Ratio Na/K vs Médicament",
                "Pression Artérielle vs Médicament",
                "Cholestérol vs Médicament",
                "Sexe vs Médicament"
            ])
            
            if vis_option == "Âge vs Médicament":
                fig = px.box(df, x="Drug", y="Age", title="Âge par Type de Médicament")
                st.plotly_chart(fig)
            
            elif vis_option == "Ratio Na/K vs Médicament":
                fig = px.box(df, x="Drug", y="Na_sur_K", title="Ratio Na/K par Type de Médicament")
                st.plotly_chart(fig)
            
            elif vis_option == "Pression Artérielle vs Médicament":
                crosstab = pd.crosstab(df_raw['BP'], df_raw['Drug'])
                fig, ax = plt.subplots(figsize=(10, 6))
                crosstab.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title("Pression Artérielle vs Médicament")
                ax.set_xlabel("Pression Artérielle")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)
            
            elif vis_option == "Cholestérol vs Médicament":
                crosstab = pd.crosstab(df_raw['Cholesterol'], df_raw['Drug'])
                fig, ax = plt.subplots(figsize=(10, 6))
                crosstab.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title("Cholestérol vs Médicament")
                ax.set_xlabel("Cholestérol")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)
            
            elif vis_option == "Sexe vs Médicament":
                crosstab = pd.crosstab(df_raw['Sex'], df_raw['Drug'])
                fig, ax = plt.subplots(figsize=(10, 6))
                crosstab.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title("Sexe vs Médicament")
                ax.set_xlabel("Sexe")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)
    else:
        st.warning("Veuillez charger un fichier de données d'abord.")

# Section d'entraînement du modèle
if menu == "Entraînement du Modèle":
    st.title("Entraînement du Modèle")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        if st.button("Entraîner le Modèle Random Forest"):
            # Préparer les features et la cible
            X = df.drop(['Drug'], axis=1)
            y = df['Drug']
            
            # Encoder la variable cible
            df_encoded = manual_encoder(df, ['Drug
