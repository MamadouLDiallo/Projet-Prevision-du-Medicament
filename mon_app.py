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

if menu == "Accueil":
    
    st.title("Présentation du mbembre du projet")
    # Create three columns layout
    left_column, middle1_column, middle2_column, right_column = st.columns(4)

# Left column - Email
    left_column.subheader("Nom")
    left_column.markdown("**Mamadou Lamarana Diallo**")
    
# middle1 column - Email
    middle1_column.subheader("📧 Email")
    middle1_column.markdown("[mamadoulamaranadiallomld1@gmail.com](mailto:mamadoulamaranadiallomld1@gmail.com)")

# Middle2 column - Phone
    middle2_column.subheader("☎️ Contact ")
    middle2_column.markdown("[+221 771050342](tel:+221771050342)")

# Right column - Linkedin
    right_column.markdown("""<h3><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align: middle;"> LinkedIn</h3> """, unsafe_allow_html=True)
    right_column.markdown("[Linkedin](https://www.linkedin.com/in/mamadou-lamarana-diallo-937430274/)")

# Section de chargement des données
if menu == "Chargement des Données":
    
    #st.title("Chargement des Données")
    
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
                crosstab.plot(kind='bar', stacstked=True, ax=ax)
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
            df_encoded = manual_encoder(df, ['Drug'])
            y_encoded = df_encoded['Drug']
            
            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Mettre à l'échelle les features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            
            # Entraîner le modèle
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            
            # Évaluation du modèle
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.subheader("Performance du Modèle")
            st.write(f"Précision : {accuracy:.3f}")
            
            # Rapport de classification
            st.subheader("Rapport de Classification")
            drug_labels = st.session_state.drug_labels
            report = classification_report(y_test, y_pred, target_names=drug_labels, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            # Matrice de confusion
            st.subheader("Matrice de Confusion")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=drug_labels, yticklabels=drug_labels)
            plt.xlabel('Prédit')
            plt.ylabel('Vrai')
            st.pyplot(fig)
            
            # Scores d'entraînement et de test
            st.subheader("Scores d'Entraînement et de Test")
            results = pd.DataFrame({
                'nom': ['RandomForest'],
                'train_score': [model.score(scaler.transform(X_train), y_train) * 100],
                'test_score': [model.score(scaler.transform(X_test), y_test) * 100]
            })
            fig, ax = plt.subplots(figsize=(5, 3))
            results.set_index('nom')[["train_score", "test_score"]].plot.bar(ax=ax)
            ax.set_title("Scores d'Entraînement et de Test (%)")
            ax.set_ylabel("Score (%)")
            st.pyplot(fig)
    else:
        st.warning("Veuillez charger un fichier de données d'abord.")

# Section de prédiction pour un patient
if menu == "Prédiction pour Patient":
    st.title("Le type de Médicament à prescrire pour un nouveau Patient")
    if st.session_state.model is not None and st.session_state.X_train is not None and st.session_state.drug_labels is not None:
        st.sidebar.header("Informations du Patient")
        
        # Formulaire d'entrée
        new_data = {}
        for col in st.session_state.X_train.columns:
            if col == 'BP':
                new_data[col] = st.sidebar.selectbox("Pression Artérielle :", ['LOW', 'NORMAL', 'HIGH'])
            elif col == 'Cholesterol':
                new_data[col] = st.sidebar.selectbox("Cholestérol :", ['NORMAL', 'HIGH'])
            elif col == 'Age':
                new_data[col] = st.sidebar.number_input("Âge :", min_value=1, max_value=100, value=30)
            #elif col == 'Na':
                #new_data[col] = st.sidebar.number_input("Sodium (Na) [mmol/L] :", min_value=100.0, max_value=160.0, value=140.0, step=0.5)
            #elif col == 'K':
                #new_data[col] = st.sidebar.number_input("Potassium (K) [mmol/L] :", min_value=2.0, max_value=7.0, value=4.0, step=0.1)
             
            elif col == 'Na':
                    new_data[col] = st.sidebar.number_input("Sodium (Na) :", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            elif col == 'K':
                new_data[col] = st.sidebar.number_input("Potassium (K) :", min_value=0.0, max_value=0.1, value=0.05, step=0.001)
        # Calculer le ratio Na/K
        new_data['Na_sur_K'] = new_data['Na'] / new_data['K']
        
        # Créer un DataFrame
        new_data_df = pd.DataFrame([new_data])
        new_data_df = manual_encoder(new_data_df, ['BP', 'Cholesterol'])
        
        # Assurer l'ordre des colonnes
        new_data_df = new_data_df[st.session_state.X_train.columns]
        
        if st.sidebar.button("Prédire le Médicament"):
            # Mettre à l'échelle les données
            new_data_scaled = st.session_state.scaler.transform(new_data_df)
            
            # Faire la prédiction
            prediction = st.session_state.model.predict(new_data_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(new_data_scaled)[0]
            
            # Décoder la prédiction
            predicted_drug = st.session_state.drug_labels[prediction]
            
            st.subheader("Résultats de la Prédiction")
            st.write(f"Médicament Recommandé : **{predicted_drug}**")
            st.write("Probabilités de Prédiction :")
            drug_labels = st.session_state.drug_labels
            for drug, prob in zip(drug_labels, prediction_proba):
                st.write(f"{drug}: {prob:.3f}")
    else:
        st.warning("Veuillez entraîner le modèle d'abord dans la section Entraînement du Modèle.")
