import streamlit as st
import requests
import json
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prédicteur de Prix Immobiliers",
    layout="wide"
)

st.title("Prédiction de prix de maison")
st.markdown("Estimez le prix d'une maison en utilisant différents modèles")

API_URL = "http://api:8000"

@st.cache_data
def get_feature_names():
    try:
        response = requests.get(f"{API_URL}/features")
        if response.status_code == 200:
            return response.json()["features"]
        else:
            return []
    except:
        try:
            return joblib.load('models/feature_names.pkl')
        except:
            return [f"feature_{i}" for i in range(10)]

@st.cache_data
def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        else:
            return ["random_forest", "xgboost", "linear_regression"]
    except:
        return ["random_forest", "xgboost", "linear_regression"]

st.sidebar.header("Configuration")
models = get_available_models()
selected_model = st.sidebar.selectbox("Modèle à utiliser", models)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Entrée des caractéristiques")

    feature_names = get_feature_names()

    if st.button("Générer des données aléatoires", type="secondary"):
        import random
        test_values = []
        for i, feature in enumerate(feature_names):
            feature_lower = feature.lower()
            if 'bedroom' in feature_lower or 'bed' in feature_lower:
                val = random.randint(2, 5)
            elif 'bathroom' in feature_lower or 'bath' in feature_lower:
                val = random.randint(1, 4)
            elif 'floor' in feature_lower:
                val = random.choice([1, 1.5, 2, 2.5, 3])
            elif 'sqft' in feature_lower or 'area' in feature_lower or 'size' in feature_lower:
                if 'lot' in feature_lower:
                    val = random.randint(2000, 15000)
                else:
                    val = random.randint(800, 4000)
            elif 'year' in feature_lower or 'built' in feature_lower:
                val = random.randint(1950, 2023)
            elif 'age' in feature_lower:
                val = random.randint(1, 70)
            elif 'grade' in feature_lower or 'condition' in feature_lower:
                val = random.randint(3, 12)
            elif 'view' in feature_lower:
                val = random.randint(0, 4)
            elif 'waterfront' in feature_lower:
                val = random.choice([0, 1])
            else:
                val = round(random.uniform(0.5, 10), 2)

            st.session_state[f"feature_{i}"] = val

        st.success("Données générées.")

    features_values = []
    cols = st.columns(2)

    for i, feature in enumerate(feature_names):
        col_idx = i % 2
        with cols[col_idx]:
            default_value = st.session_state.get(f"feature_{i}", 0.0)
            value = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=float(default_value),
                key=f"feature_{i}"
            )
            features_values.append(value)

with col2:
    st.header("Résultat")

    if st.button("Prédire", type="primary"):
        prediction_data = {
            "features": features_values,
            "model_name": selected_model
        }

        try:
            with st.spinner("Chargement..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=prediction_data
                )

            if response.status_code == 200:
                result = response.json()
                st.success("Prédiction réussie")
                st.metric(
                    label="Prix estimé",
                    value=f"${result['prediction']:,.2f}"
                )
                st.caption(f"Modèle utilisé : {result['model_used']}")
                st.caption(f"Heure : {result['timestamp']}")
            else:
                error_detail = response.json().get("detail", "Erreur inconnue")
                st.error(f"Erreur : {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter à l'API.")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
    
try:
    health_response = requests.get(f"{API_URL}/loggers", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("API connectée")
    else:
        st.sidebar.error("API non disponible")
except:
    st.sidebar.error("API non disponible")
