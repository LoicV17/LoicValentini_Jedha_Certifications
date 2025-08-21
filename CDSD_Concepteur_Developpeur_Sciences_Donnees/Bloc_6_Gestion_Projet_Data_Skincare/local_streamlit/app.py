import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="Skin Care - Analyse des grains de beauté",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from PIL import Image

# Preprocess spécifiques aux modèles
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

import joblib
import plotly.graph_objects as go
from GradCam import generate_gradcam  # suppose gradcam basé sur model2

# ============== Chargement des modèles ==============
MODEL1_PATH = "model1.keras"     # EfficientNetB0 binaire (benin/malin)
MODEL2_PATH = "model2.h5"        # ResNet/Xception multiclasses (7 classes)
MODEL3_PATH = "model3.joblib"    # pipeline tabulaire (stacked RF)

@st.cache_resource(show_spinner=True)
def load_model1():
    return tf.keras.models.load_model(MODEL1_PATH)

@st.cache_resource(show_spinner=True)
def load_model2():
    return tf.keras.models.load_model(MODEL2_PATH)

@st.cache_resource(show_spinner=True)
def load_model3():
    return joblib.load(MODEL3_PATH)

model1 = load_model1()
model2 = load_model2()
model3 = load_model3()

# ============== Métadonnées classes ==============
classes = {
    0: 'akiec - kératoses actiniques',
    1: 'bcc - carcinome basocellulaire',
    2: 'bkl - kératoses séborrhéiques',
    3: 'df - dermatofibromes',
    4: 'mel - melanoma',
    5: 'nv - névus mélanocytaire',
    6: 'vasc - lésions vasculaires'
}

CODE_TO_FULL = {code: label for code, label in [
    ('akiec', 'akiec - kératoses actiniques'),
    ('bcc',   'bcc - carcinome basocellulaire'),
    ('bkl',   'bkl - kératoses séborrhéiques'),
    ('df',    'df - dermatofibromes'),
    ('mel',   'mel - melanoma'),
    ('nv',    'nv - névus mélanocytaire'),
    ('vasc',  'vasc - lésions vasculaires'),
]}

CLASS_EXPLANATIONS = {
    0: "lésion précancéreuse liée au soleil (à surveiller)",
    1: "cancer cutané à croissance lente (consultation recommandée)",
    2: "lésion bénigne fréquente",
    3: "petit nodule bénin fibreux",
    4: "cancer de la peau potentiellement dangereux (urgence médicale)",
    5: "grain de beauté standard (le plus souvent bénin)",
    6: "lésion des vaisseaux (ex. angiome, hémangiome)",
}

# ============== Préprocess robustes ==============
def preprocess_image_for_model1_from_pil(pil_img, target_size=(240, 240)):
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    arr = tf.image.resize(arr, target_size, method="bilinear")
    arr = tf.cast(arr, tf.float32).numpy()
    arr = np.expand_dims(arr, axis=0)
    arr = efficientnet_preprocess(arr)
    return arr

def preprocess_image_for_model2_from_pil(pil_img, target_size=(224, 224)):
    # si ton modèle2 est Xception, passe à (299,299) + xception_preprocess
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    arr = tf.image.resize(arr, target_size, method="bilinear")
    arr = tf.cast(arr, tf.float32).numpy()
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)
    return arr

# ============== Prédiction unifiée ==============
def predict_image(pil_image, age, sex, localization, use_tabular=True):
    # --- modèle 1 (bénin/malin) ---
    img_array_model1 = preprocess_image_for_model1_from_pil(pil_image, target_size=(240, 240))
    result_model1 = model1.predict(img_array_model1)[0][0]
    proba_malin_pct = float(np.round(result_model1 * 100, 1))

    # --- modèle 2 (7 classes) ---
    img_array_model2 = preprocess_image_for_model2_from_pil(pil_image, target_size=(224, 224))
    result_model2 = model2.predict(img_array_model2)[0]
    top_idx = int(np.argmax(result_model2))
    predicted_full_label = classes[top_idx]  # fallback image-only

    # Top-3 texte avec explications
    top_3_idx = np.argsort(result_model2)[::-1][:3]
    top_3_lines = []
    for i in top_3_idx:
        pct = result_model2[i] * 100
        expl = CLASS_EXPLANATIONS.get(i, "")
        top_3_lines.append(f"{classes[i]} : {pct:.1f}% — {expl}")
    top_3_text = "\n".join(top_3_lines)

    # --- Grad-CAM (toujours activée) ---
    gradcam_image = generate_gradcam(
        pil_image, model=model2, preprocess_fn=resnet_preprocess, target_size=(224, 224),
    )

    # --- modèle 3 (stacked RF) si use_tabular ---
    result_model3 = None
    if use_tabular:
        features = {
            "age": age, "sex": sex, "localization": localization,
            "proba_akiec": float(result_model2[0]),
            "proba_bcc":   float(result_model2[1]),
            "proba_bkl":   float(result_model2[2]),
            "proba_df":    float(result_model2[3]),
            "proba_mel":   float(result_model2[4]),
            "proba_nv":    float(result_model2[5]),
            "proba_vasc":  float(result_model2[6]),
        }
        input_df = pd.DataFrame([features])
        result_model3 = model3.predict(input_df)[0]
        # normalisation d’affichage
        if isinstance(result_model3, str):
            code = result_model3.strip().split()[0].split('-')[0]
            predicted_full_label = CODE_TO_FULL.get(code, result_model3)
        elif isinstance(result_model3, (int, np.integer)) and int(result_model3) in classes:
            predicted_full_label = classes[int(result_model3)]
        else:
            predicted_full_label = str(result_model3)

    return gradcam_image, proba_malin_pct, result_model2, top_3_text, result_model3, predicted_full_label

# ============== UI Streamlit ==============
st.markdown("<h1 style='text-align: center;'>🔎 Skin Care - Analyse des grains de beauté 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Soumettez une image et obtenez une prédiction du caractère bénin/malin, ainsi qu'une classification dermatologique.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# (optionnel) style
st.markdown(
    """
    <style>
        .css-ffhzg2 { border-left: 2px solid black; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Colonne gauche -----------------
with col1:
    st.subheader("📥 Import manuel ou Webcam")

    camera_image = st.camera_input("Prenez une photo")
    uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.subheader("📁 Ou utilisez un exemple")
    example_files = ["Exemple1.jpg", "Exemple2.jpg", "Exemple3.jpg", "Exemple4.jpg", "Exemple5.jpg", "Exemple6.jpg"]
    selected_example = st.selectbox("Choisissez un exemple :", ["-- Aucun --"] + example_files)

    st.markdown("---")
    st.subheader("👤 Informations Patient")
    missing_patient_info = st.checkbox("Je n'ai pas toutes les informations patient")
    age = st.slider("Âge", 0, 100, 5, disabled=missing_patient_info)
    sex = st.selectbox("Sexe", ["male", "female"], disabled=missing_patient_info)
    localization = st.selectbox(
        "Localisation de la lésion - choisir le plus proche",
        ["scalp", "ear", "face", "back", "chest", "trunk", "upper extremity",
         "lower extremity", "genital", "abdomen", "unknown"],
        disabled=missing_patient_info
    )

    # Chargement de l'image — ⚠️ on NE l'affiche PLUS dans la colonne gauche
    input_image = None
    if camera_image is not None:
        input_image = Image.open(camera_image).convert("RGB")
    elif uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
    elif selected_example != "-- Aucun --":
        image_path = os.path.join("examples", selected_example)
        if os.path.exists(image_path):
            input_image = Image.open(image_path).convert("RGB")

    st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- Colonne droite -----------------
with col2:
    if input_image is not None:
        gradcam_image, proba_malin, result_model2, top_3_text, result_model3, predicted_full_label = predict_image(
            input_image,
            age if not missing_patient_info else None,
            sex if not missing_patient_info else None,
            localization if not missing_patient_info else None,
            use_tabular=(not missing_patient_info)
        )

        # ====== 1) Votre photo ======
        st.subheader("📸 Votre photo")
        display_img = input_image.copy()
        try:
            display_img = display_img.resize((600, 400))  # affichage uniquement
        except Exception:
            pass
        st.image(display_img, width=600)
        st.markdown("---")

        # ====== 2) Résultat général (type de lésion + phrase de risque) ======
        st.subheader("🧾 Résultat général")

        # Code de la classe prédite (issu du modèle 3 si dispo, sinon top-1 du modèle 2)
        predicted_code = predicted_full_label.split(" - ")[0].strip().lower()

        def calculate_risk(proba_malin_pct, probs_model2, predicted_code: str):
            """
            Règles combinées :
            - ÉLEVÉ si (p_bin > 50) OU (s_mal > 0.50) OU (classe ∈ {akiec, bcc, mel})
            - FAIBLE si (p_bin < 10) ET (s_mal < 0.10) ET (classe ∈ {bkl, df, nv, vasc})
            - Sinon MODÉRÉ
            Où:
            p_bin = proba_malin_pct (modèle 1 binaire)  [0..100]
            s_mal = somme des probas modèle 2 pour {akiec(0), bcc(1), mel(4)}  [0..1]
            """
            malignant_idxs = [0, 1, 4]  # akiec, bcc, mel
            malignant_codes = {"akiec", "bcc", "mel"}
            benign_codes    = {"bkl", "df", "nv", "vasc"}

            s_mal = float(np.sum(probs_model2[malignant_idxs]))  # 0..1

            is_high = (proba_malin_pct > 50) or (s_mal > 0.50) or (predicted_code in malignant_codes)
            is_low  = (proba_malin_pct < 10) and (s_mal < 0.10) and (predicted_code in benign_codes)

            if is_high:
                return ("Risque élevé", "red",
                        "Notre application a détecté un risque élevé. Nous vous recommandons de prendre un rendez-vous aussi vite que possible chez un professionnel de santé.",
                        s_mal)
            elif is_low:
                return ("Risque faible", "green",
                        "Le risque détecté est faible, mais il est toujours recommandé de surveiller vos grains de beauté régulièrement.",
                        s_mal)
            else:
                return ("Risque modéré", "orange",
                        "Le risque est modéré. Il est conseillé de consulter un professionnel de santé pour un suivi.",
                        s_mal)

        risk_text, risk_color, risk_message, s_mal = calculate_risk(proba_malin, result_model2, predicted_code)

        # Palette & pictogrammes selon le risque
        risk_palette = {
            "green":  {"bg": "#ecfdf5", "border": "#10b981", "title": "#065f46", "text": "#065f46", "icon": "🟢"},
            "orange": {"bg": "#fff7ed", "border": "#f59e0b", "title": "#7c2d12", "text": "#7c2d12", "icon": "🟠"},
            "red":    {"bg": "#fef2f2", "border": "#ef4444", "title": "#7f1d1d", "text": "#7f1d1d", "icon": "🔴"},
        }
        palette = risk_palette.get(risk_color, risk_palette["orange"])
        icon = palette["icon"]

        # Box unifiée : type de lésion + phrase de risque + note sur l’échelle
        st.markdown(
            f"""
            <div style='
                background-color: {palette["bg"]};
                border-left: 6px solid {palette["border"]};
                padding: 16px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            '>
                <div style='font-size: 18px; color: {palette["title"]}; font-weight: 700;'>
                    Type de lésion le plus probable : {predicted_full_label}
                </div>
                <div style='margin-top: 6px; color: {palette["text"]}; font-size: 16px;'>
                    <strong>{icon} {risk_text}</strong> — {risk_message}
                </div>
                <div style='margin-top: 8px; font-size: 13px; color: #6b7280;'>
                    ℹ️ L’échelle de risque comporte uniquement trois niveaux&nbsp;: <em>faible</em>, <em>modéré</em> et <em>élevé</em>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Si infos patient manquantes, on l’indique en dehors de la box
        if result_model3 is None:
            st.info("Mode sans métadonnées : estimation basée sur l'image uniquement (modèle 2).")

        st.markdown("---")


        # ====== 3) Jauge ======
        st.subheader("🩺 Jauge de probabilité bénin / malin")
        def get_color(proba):
            if proba < 10:  return "#6EE7B7"
            if proba < 30:  return "#FDE68A"
            if proba < 50:  return "#FDC78A"
            if proba < 70:  return "#FCA5A5"
            return "#EF4444"

        rounded_proba = float(np.round(proba_malin, 0))
        score_color = get_color(rounded_proba)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rounded_proba,
            number={'font': {'color': score_color}},
            title={'text': "Risque malin (%) — modèle binaire"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'rgba(0,0,0,0)'},
                'steps': [
                    {'range': [0, 10],  'color': "#6EE7B7"},
                    {'range': [10, 30], 'color': "#FDE68A"},
                    {'range': [30, 50], 'color': "#FDC78A"},
                    {'range': [50, 100],'color': "#EF4444"},
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

        # ====== 4) Diagnostic Top 3 ======
        st.subheader("🔍 Diagnostic — Top 3 classes — modèle multi-classe")
        top3_rows = []
        for i in np.argsort(result_model2)[::-1][:3]:
            top3_rows.append({
                "Code": classes[i].split(" - ")[0],
                "Lésion": classes[i].split(" - ")[1],
                "Probabilité": f"{result_model2[i]*100:.1f} %",
                "Explication": CLASS_EXPLANATIONS.get(i, "")
            })
        st.table(pd.DataFrame(top3_rows))

        # Exemples d’images pour classes >10%
        st.markdown("#### 📸 Exemples des classes détectées (>10%)")
        high_proba = sorted(
            [(idx, p) for idx, p in enumerate(result_model2) if p > 0.10],
            key=lambda x: x[1], reverse=True
        )
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

        # ====== 5) Grad-CAM ======
        st.subheader("🧠 Grad-CAM — Zones d'attention du modèle")
        centered_col = st.columns([1, 2, 1])[1]
        with centered_col:
            st.image(gradcam_image, width=300)
        st.markdown("---")

        # ====== 6) Conseils ======
        st.subheader("💡 Conseils Skincare")
        st.write(
            "💡 Ce modèle vous donne un aperçu du risque associé à l’image et propose une classification dermatologique automatisée.<br> "
            "👨‍⚕️ Cette application ne remplace en aucun cas l'avis d'un professionnel de santé.<br>"
            "👩‍⚕️ Consultez un dermatologue en cas de doute ou de changement rapide.<br>"
            "🔆 Appliquez une crème solaire à large spectre tous les jours, même en hiver.<br>"
            "📅 Surveillez vos grains de beauté tous les 3 mois (ABCD : Asymétrie, Bords, Couleur, Diamètre).<br>"
            "🧴 Choisissez des produits de soin adaptés (peau sèche, grasse, sensible, etc.).<br>"
            "💧 Hydratez régulièrement pour maintenir une barrière cutanée saine.",
            unsafe_allow_html=True
        )
    else:
        st.info("Importez ou prenez une photo pour lancer l'analyse.")
