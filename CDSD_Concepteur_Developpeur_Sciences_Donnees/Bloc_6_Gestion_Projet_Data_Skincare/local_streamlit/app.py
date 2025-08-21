import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Preprocess spécifiques aux modèles
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

import joblib
import plotly.graph_objects as go
from GradCam import generate_gradcam  # suppose gradcam basé sur model2 (Xception)

# ============== Chargement des modèles ==============
# Assure-toi que ces 3 fichiers sont bien dans le même dossier que app.py
MODEL1_PATH = "model1.keras"           # EfficientNetB0 binaire (benin/malin)
MODEL2_PATH = "model2.h5"           # Xception multiclasses
MODEL3_PATH = "model3_full_pipeline.pkl"

model1 = tf.keras.models.load_model(MODEL1_PATH)
model2 = tf.keras.models.load_model(MODEL2_PATH)
model3 = joblib.load(MODEL3_PATH)

# ============== Métadonnées classes (model2) ==============
classes = {
    0: 'akiec - kératoses actiniques',
    1: 'bcc - carcinome basocellulaire',
    2: 'bkl - kératoses séborrhéiques',
    3: 'df - dermatofibromes',
    4: 'mel - melanoma',
    5: 'nv - névus mélanocytaire',
    6: 'vasc - lésions vasculaires'
}

# ============== Préprocess robustes ==============
def preprocess_image_for_model1_from_pil(pil_img, target_size=(240, 240)):
    """
    EfficientNetB0 (model1) — entraîné avec efficientnet_preprocess.
    Force RGB, resize via TF, preprocess officiel, batch dim.
    """
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)                      # (H, W, 3)
    arr = tf.image.resize(arr, target_size, method="bilinear")
    arr = tf.cast(arr, tf.float32).numpy()
    arr = np.expand_dims(arr, axis=0)            # (1, H, W, 3)
    arr = efficientnet_preprocess(arr)           # EXACTEMENT comme au training
    return arr

def preprocess_image_for_model2_from_pil(pil_img, target_size=(224, 224)):
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    arr = tf.image.resize(arr, target_size, method="bilinear")
    arr = tf.cast(arr, tf.float32).numpy()
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)   # 🔑 preprocess ResNet50
    return arr


# ============== Prédiction unifiée ==============
def predict_image(pil_image, age, sex, localization):
    # --- Modèle 1 : Prédiction bénin/malin ---
    img_array_model1 = preprocess_image_for_model1_from_pil(pil_image, target_size=(240, 240))
    result_model1 = model1.predict(img_array_model1)[0][0]
    proba_malin = round(result_model1 * 100, 1)  # 1 décimale

    # --- Modèle 2 : Classification dermatologique ---
    img_array_model2 = preprocess_image_for_model2_from_pil(pil_image, target_size=(224, 224))
    result_model2 = model2.predict(img_array_model2)[0]  # shape (7,)

    # Top 3 classes
    top_3_idx = np.argsort(result_model2)[::-1][:3]
    top_3_text = "\n".join([f"{classes[i]} : {result_model2[i] * 100:.1f}%" for i in top_3_idx])

    # --- Modèle 3 : Diagnostic combiné (pipeline tabulaire) ---
    features = {
        "age": age,
        "sex": sex,
        "localization": localization,
        "proba_malign": float(result_model1),  # proba brute 0–1 pour le pipeline
        "akiec": float(result_model2[0]),
        "bcc":  float(result_model2[1]),
        "bkl":  float(result_model2[2]),
        "df":   float(result_model2[3]),
        "nv":   float(result_model2[5]),
        "vasc": float(result_model2[6]),
        "mel":  float(result_model2[4])
    }
    input_df = pd.DataFrame([features])
    result_model3 = model3.predict(input_df)[0]

    gradcam_image = generate_gradcam(
        pil_image,                 # pas besoin de resize ici, la fonction s'en charge
        model=model2,
        preprocess_fn=resnet_preprocess,
        target_size=(224, 224),
    # last_conv_layer_name="block14_sepconv2_act"  # ← tu peux préciser, sinon commente pour auto
)

    return gradcam_image, proba_malin, result_model2, top_3_text, result_model3

# ============== UI Streamlit ==============
st.set_page_config(layout="wide")

# Titre
st.markdown("<h1 style='text-align: center;'>🔎 Skin Care - Analyse des grains de beauté 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Soumettez une image et obtenez une prédiction du caractère bénin/malin, ainsi qu'une classification dermatologique.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# Petite ligne de style (optionnelle)
st.markdown(
    """
    <style>
        .css-ffhzg2 {
            border-left: 2px solid black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

image = None

with col1:
    st.subheader("📥 Import manuel ou Webcam")

    camera_image = st.camera_input("Prenez une photo")
    uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.subheader("📁 Ou utilisez un exemple")
    example_files = ["Exemple1.jpg", "Exemple2.jpg", "Exemple3.jpg", "Exemple4.jpg", "Exemple5.jpg", "Exemple6.jpg"]
    selected_example = st.selectbox("Choisissez un exemple :", ["-- Aucun --"] + example_files)

    st.subheader("👤 Informations Patient")
    age = st.slider("Âge", 0, 100, 5)
    sex = st.selectbox("Sexe", ["male", "female"])
    localization = st.selectbox(
        "Localisation de la lésion - choisir le plus proche",
        ["scalp", "ear", "face", "back", "chest", "trunk", "upper extremity",
         "lower extremity", "genital", "abdomen", "unknown"]
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Photo capturée via la webcam", use_column_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image importée", use_column_width=True)
    elif selected_example != "-- Aucun --":
        image_path = os.path.join("examples", selected_example)
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption=f"Exemple : {selected_example}", use_column_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

with col2:
    if image is not None:
        gradcam_image, proba_malin, result_model2, top_3_text, result_model3 = predict_image(image, age, sex, localization)

        # ====== Section 1 : Résultat global ======
        st.markdown("### 🧾 Résultat global")

        def calculate_risk(proba_malin_pct, probs_model2):
            # proba_malin_pct est en %
            risk_high = (proba_malin_pct > 50) or (np.sum(probs_model2[[0, 1, 4]]) > 0.30)
            risk_low  = (proba_malin_pct < 11) and (np.sum(probs_model2[[0, 1, 4]]) < 0.10)
            if risk_high:
                return ("Risque élevé", "red",
                        "Notre application a détecté un risque élevé. Nous vous recommandons de prendre un rendez-vous aussi vite que possible chez un professionnel de santé, médecin traitant ou dermatologue.")
            elif risk_low:
                return ("Risque faible", "green",
                        "Le risque détecté est faible, mais il est toujours recommandé de surveiller vos grains de beauté régulièrement.")
            else:
                return ("Risque modéré", "orange",
                        "Le risque est modéré. Il est conseillé de consulter un professionnel de santé pour un suivi, surtout si des changements sont observés.")

        risk_text, risk_color, risk_message = calculate_risk(proba_malin, result_model2)
        st.markdown(f"#### <span style='font-size: 30px; color: {risk_color};'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: {risk_color};'>{risk_message}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # ====== Section 2 : Jauge ======
        st.markdown("### 🩺 Jauge de probabilité bénin / malin")

        def get_color(proba):
            if proba < 20:  return "#6EE7B7"
            if proba < 40:  return "#A7F3D0"
            if proba < 60:  return "#FDE68A"
            if proba < 80:  return "#FCA5A5"
            return "#EF4444"

        rounded_proba = float(np.round(proba_malin, 1))
        score_color = get_color(rounded_proba)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rounded_proba,
            number={'font': {'color': score_color}},
            title={'text': "Risque malin (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'rgba(0,0,0,0)'},
                'steps': [
                    {'range': [0, 20],  'color': "#6EE7B7"},
                    {'range': [20, 40], 'color': "#A7F3D0"},
                    {'range': [40, 60], 'color': "#FDE68A"},
                    {'range': [60, 80], 'color': "#FCA5A5"},
                    {'range': [80, 100],'color': "#EF4444"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': rounded_proba
                }
            }
        ))
        st.plotly_chart(fig, use_column_width=True)

        st.markdown("---")

        # ====== Section 3 : Diagnostic + Top3 ======
        st.markdown("### 🔍 Diagnostic")
        st.markdown(f"""
            <div style='
                background-color: #fcebea;
                border-left: 6px solid darkred;
                padding: 16px;
                margin: 10px 0 20px 0;
                border-radius: 8px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            '>
                <h3 style='color: darkred; margin: 0;'>🧾 Prédiction de lésion : {result_model3}</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:16px; white-space: pre-wrap;'>{top_3_text}</div>", unsafe_allow_html=True)

        # Exemples d’images pour classes >10%
        st.markdown("#### 📸 Exemples des classes détectées (>10%)")
        high_proba = [(idx, p) for idx, p in enumerate(result_model2) if p > 0.10]
        if len(high_proba) > 0:
            cols = st.columns(len(high_proba))
            for col, (idx, p) in zip(cols, high_proba):
                class_code = classes[idx].split(' - ')[0]
                class_label = classes[idx]
                image_path = os.path.join("classes", f"{class_code}.jpg")
                if os.path.exists(image_path):
                    with col:
                        st.image(image_path, caption=f"{class_label} ({p*100:.0f}%)", width=200)
        else:
            st.write("Aucune classe > 10%.")

        st.markdown("---")

        # ====== Section 4 : Grad-CAM ======
        st.markdown("### 🧠 Visualisation Grad-CAM / Zones qui ont impacté l'analyse")
        centered_col = st.columns([1, 2, 1])[1]
        with centered_col:
            st.image(gradcam_image, width=300)
        st.markdown("---")

        # ====== Section 5 : Conseils ======
        st.markdown("### 💡 Conseils Skincare")
        st.write(
            "💡 Ce modèle vous donne un aperçu du risque associé à l’image et propose une classification dermatologique automatisée.<br> "
            "👨‍⚕️ Cette application ne remplace en aucun cas l'avis d'un professionnel de santé.<br>"
            "👩‍⚕️ Consultez un dermatologue en cas de doute ou de changement rapide.<br>"
            "🔆 Appliquez une crème solaire à large spectre tous les jours, même en hiver.<br>"
            "📅 Surveillez vos grains de beauté tous les 3 mois (ABCD : Asymétrie, Bords, Couleur, Diamètre).<br>"
            "🧴 Choisissez des produits de soin adaptés à votre type de peau et à vos besoins spécifiques (peau sèche, grasse, sensible, etc.).<br>"
            "💧 Hydratez votre peau régulièrement avec des crèmes et sérums adaptés pour maintenir une barrière cutanée saine.<br>"
            "🚶‍♂️ Évitez une exposition excessive au soleil, surtout entre 12h et 16h, lorsque les rayons UV sont les plus forts.<br>"
            "🧑‍⚕️ Si vous remarquez un changement dans un grain de beauté (forme, couleur, taille), consultez immédiatement un professionnel de santé.<br>"
            "🍏 Adoptez une alimentation équilibrée riche en antioxydants (fruits, légumes, acides gras essentiels) pour soutenir la santé de votre peau.",
            unsafe_allow_html=True
        )
