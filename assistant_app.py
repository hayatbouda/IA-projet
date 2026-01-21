import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model 
import time
import base64 
import io

st.set_page_config(page_title="IA Sanguine")

@st.cache_data
def get_base64_of_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.sidebar.error(f"❌ Image '{path}' non trouvée! Assurez-vous qu'elle est dans le dossier.")
        return None

def set_background(base64_img):
    if base64_img:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_img}"); 
            background-size: cover; 
            background-attachment: fixed; 
        }}
        label {{
            color: black;
            font-weight: bold;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

base64_image = get_base64_of_image('sang.webp') 
set_background(base64_image)

try:
    scaler_t = joblib.load('scaler_transfusion.pkl')
    modele_t = joblib.load('modele_transfusion.pkl')
    
    scaler_a = joblib.load('scaler_anemie.pkl')
    modele_anemie = joblib.load('modele_anemie.pkl')
    modele_urgence = joblib.load('modele_urgence.pkl')
    
    st.sidebar.success("✅ Tous les modèles de l'Assistant ont été chargés.")
except FileNotFoundError:
    st.sidebar.error("❌ ERREUR DE CHARGEMENT: Assurez-vous d'avoir exécuté les Étapes 1, 2 et 3.")
    st.stop()

full_title = "🩺 Assistant Médical Intelligent - Analyse Sanguine"
title_placeholder = st.empty() 
char_delay = 0.05 

for i in range(len(full_title) + 1):
    html_content = f"<h1 style='color: #1976D2; font-weight: bold;'>{full_title[:i]}</h1>"
    title_placeholder.markdown(html_content, unsafe_allow_html=True)
    time.sleep(char_delay)

st.markdown("---") 

st.markdown('<h2 style="color: black; font-weight: bold;">📝 Saisie des Données du Patient</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations de Don/Transfusion")
    recency = st.number_input('Récence (mois depuis dernier don)', min_value=0, max_value=75, value=20)
    frequency = st.number_input('Fréquence (nombre de dons)', min_value=1, max_value=60, value=5)
    monetary = st.number_input('Volume Total Donné (cc)', min_value=250, max_value=15000, value=1250)
    time_input = st.number_input('Temps (mois depuis premier don)', min_value=0, max_value=75, value=40) 

with col2:
    st.subheader("Résultats d'Analyses Sanguines (Hématologie)")
    hemoglobine = st.number_input('Taux d\'Hémoglobine (g/dL)', min_value=5.0, max_value=20.0, value=13.0, step=0.1)
    mch = st.number_input('MCH (pg)', min_value=15.0, max_value=40.0, value=28.0, step=0.1)
    mcv = st.number_input('MCV (fL)', min_value=70.0, max_value=110.0, value=90.0, step=0.1)

submit_button = st.button('Analyser & Prédire', type="primary")

if submit_button:
    st.markdown("---")
    st.header("✨ Résultats de l'Assistant")

    input_t = np.array([[recency, frequency, monetary, time_input]]) 
    input_t_scaled = scaler_t.transform(input_t)
    
    input_a = np.array([[hemoglobine, mch, mcv]])
    input_a_scaled = scaler_a.transform(input_a)

    proba_t = modele_t.predict_proba(input_t_scaled)[0][1] * 100
    
    pred_anemie = modele_anemie.predict(input_a_scaled)[0]
    proba_anemie = modele_anemie.predict_proba(input_a_scaled)[0][1] * 100
    
    pred_urgence = modele_urgence.predict(input_a_scaled)[0]

    col_t, col_a, col_u = st.columns(3)

    with col_t:
        st.subheader("1. Besoin de Transfusion")
        if proba_t > 50:
            st.error(f"Probabilité de Besoin: **{proba_t:.1f}%**")
            st.markdown("Recommandation: **Forte**")
        else:
            st.success(f"Probabilité de Besoin: **{proba_t:.1f}%**")
            st.markdown("Recommandation: **Faible**")
            
    with col_a:
        st.subheader("2. Diagnostic d'Anémie")
        if pred_anemie == 1:
            st.warning(f"Statut: **ANÉMIE** | Confiance: {proba_anemie:.1f}%")
        else:
            st.success(f"Statut: **Sain** | Confiance: {(100 - proba_anemie):.1f}%")

    with col_u:
        st.subheader("3. Niveau d'Urgence")
        if pred_urgence == 1:
            st.error("Niveau: **URGENT**")
            st.markdown("Action: **Intervention Médicale Immédiate**")
        else:
            st.info("Niveau: **Non Urgent**")
            st.markdown("Action: **Suivi et Traitement Standard**")