import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt
import io
import os
import joblib
import sklearn
import pandas as pd
from GradCam import generate_gradcam
import plotly.graph_objects as go
import plotly.express as px


# Charger les modèles
model1 = tf.keras.models.load_model("model1.h5")
model2 = tf.keras.models.load_model("model2.keras")
model3 = joblib.load("model3_full_pipeline.pkl")

# Classes pour le modèle 2
classes = {
    0: 'akiec - kératoses actiniques',
    1: 'bcc - carcinome basocellulaire',
    2: 'bkl - kératoses séborrhéiques',
    3: 'df - dermatofibromes',
    6: 'mel - melanoma',
    4: 'nv - névus mélanocytaire',
    5: 'vasc - lésions vasculaires'
}

def preprocess_image_for_model1_from_pil(pil_img, target_size):
    img = pil_img.resize(target_size)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_for_model2_from_pil(pil_img, target_size):
    img = pil_img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(pil_image, age, sex, localization):
    # --- Modèle 1 : Prédiction bénin/malin ---
    img_array_model1 = preprocess_image_for_model1_from_pil(pil_image, target_size=(224, 224))
    result_model1 = model1.predict(img_array_model1)[0][0]
    proba_malin = round(result_model1 * 100, -1)


    # --- Modèle 2 : Classification dermatologique ---
    img_array_model2 = preprocess_image_for_model2_from_pil(pil_image, target_size=(299, 299))
    result_model2 = model2.predict(img_array_model2)[0]

    # Tri des classes par probabilité décroissante et récupération des 3 meilleures
    top_3_classes_idx = np.argsort(result_model2)[::-1][:3]
    top_3_classes_prob = result_model2[top_3_classes_idx]
    top_3_text = "\n".join([
        f"{classes[i]} : {prob * 100:.1f}%"
        for i, prob in zip(top_3_classes_idx, top_3_classes_prob)
    ])

    # --- Modèle 3 : Diagnostic combiné (pipeline arbre de décision) ---
    # Créer un DataFrame avec les valeurs de l'utilisateur et les sorties des modèles
    features = {
        "age": age,  # Récupérer directement les valeurs saisies par l'utilisateur
        "sex": sex,
        "localization": localization,
        "proba_malign": result_model1,  # Sortie du modèle 1 : probabilité bénin/malin
        "akiec": result_model2[0],     # Sorties du modèle 2 : classes dermatologiques
        "bcc": result_model2[1],
        "bkl": result_model2[2],
        "df": result_model2[3],
        "nv": result_model2[4],
        "vasc": result_model2[5],
        "mel": result_model2[6]
    }
    input_df = pd.DataFrame([features])

    # Prédiction avec le modèle 3 (pipeline arbre de décision)
    result_model3 = model3.predict(input_df)[0]  # Classe prédite par le modèle 3

    # GradCAM basé sur modèle 2
    gradcam_image = generate_gradcam(pil_image.resize((224,224)))

    # Retourner les résultats
    return gradcam_image, proba_malin, result_model2, top_3_text, result_model3

# Initialize image to None
image = None

# --- Interface Streamlit ---
st.set_page_config(layout="wide")

# Titre centré
st.markdown("<h1 style='text-align: center;'>🔎 Skin Care - Analyse des grains de beauté 🔍</h1>", unsafe_allow_html=True)

# Texte centré
st.markdown("<p style='text-align: center;'>Soumettez une image et obtenez une prédiction du caractère bénin/malin, ainsi qu'une classification dermatologique.</p>", unsafe_allow_html=True)

# Ajouter un espace vide sous le titre
st.markdown("<br>", unsafe_allow_html=True)

# Ajouter une barre horizontale noire
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

# Créer deux colonnes
col1, col2 = st.columns([1, 2])

# Ajouter une barre verticale entre les colonnes
st.markdown(
    """
    <style>
        .css-ffhzg2 {  /* Cible la classe des colonnes Streamlit */
            border-left: 2px solid black;  /* Définir une barre verticale noire */
        }
    </style>
    """, unsafe_allow_html=True)

with col1:
    st.subheader("📥 Import manuel ou Webcam")

    # Ajouter le bouton pour la webcam
    camera_image = st.camera_input("Prenez une photo")

    uploaded_file = st.file_uploader("Ou choisissez une image JPG...", type="jpg")

    st.markdown("---")
    st.subheader("📁 Ou utilisez un exemple")
    example_files = ["Exemple1.jpg", "Exemple2.jpg", "Exemple3.jpg", "Exemple4.jpg", "Exemple5.jpg", "Exemple6.jpg"]
    selected_example = st.selectbox("Choisissez un exemple :", ["-- Aucun --"] + example_files)

    st.subheader("👤 Informations Patient")

    age = st.slider("Âge", 0, 100, 5)
    sex = st.selectbox("Sexe", ["male", "female"])
    localization = st.selectbox("Localisation de la lésion - choisir le plus proche", [
        "scalp", "ear", "face",
        "back", "chest", "trunk", "upper extremity", "lower extremity", "genital", "abdomen", "unknown"
    ])

    if camera_image is not None:
        # Convertir l'image de la webcam en format PIL
        image = Image.open(camera_image)
        st.image(image, caption="Photo capturée via la webcam", use_column_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée", use_column_width=True)
    elif selected_example != "-- Aucun --":
        image_path = os.path.join("examples", selected_example)
        image = Image.open(image_path)
        st.image(image, caption=f"Exemple : {selected_example}", use_column_width=True)

    # Ajouter une barre horizontale ici
    st.markdown("<hr>", unsafe_allow_html=True)

with col2:
    if image is not None:
        # Call the predict_image function only if image is successfully loaded
        gradcam_image, proba_malin, result_model2, top_3_text, result_model3 = predict_image(image, age, sex, localization)

        # Section 1 : Résultat global
        st.markdown("### 🧾 Résultat global")

        # Calcul du risque
        def calculate_risk(proba_malin, result_model2):
            risk_high = proba_malin > 50 or np.sum(result_model2[[6, 5, 3]]) > 0.30
            risk_low = proba_malin < 11 and np.sum(result_model2[[6, 5, 3]]) < 0.10

            if risk_high:
                return "Risque élevé", "red", "Notre application a détecté un risque élevé. Nous vous recommandons de prendre un rendez-vous aussi vite que possible chez un professionnel de santé, médecin traitant ou dermatologue."
            elif risk_low:
                return "Risque faible", "green", "Le risque détecté est faible, mais il est toujours recommandé de surveiller vos grains de beauté régulièrement."
            else:
                return "Risque modéré", "orange", "Le risque est modéré. Il est conseillé de consulter un professionnel de santé pour un suivi, surtout si des changements sont observés."

        # Calcul du risque à partir des prédictions
        risk_text, risk_color, risk_message = calculate_risk(proba_malin, result_model2)

        # Affichage du risque
        st.markdown(f"#### <span style='font-size: 30px; color: {risk_color};'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: {risk_color};'>{risk_message}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Section 2 : Jauge
        st.markdown("### 🩺 Jauge de probabilité bénin / malin")
        # Détermination de la couleur du texte en fonction du score
        def get_color(proba):
            if proba < 20:
                return "#6EE7B7"  # vert clair
            elif proba < 40:
                return "#A7F3D0"
            elif proba < 60:
                return "#FDE68A"
            elif proba < 80:
                return "#FCA5A5"
            else:
                return "#EF4444"  # rouge vif

        rounded_proba = int(round(proba_malin, 0))  # Arrondi à la dizaine
        score_color = get_color(rounded_proba)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rounded_proba,
            number={'font': {'color': score_color}},  # Couleur dynamique du score
            title={'text': "Risque malin (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'rgba(0,0,0,0)'},  # Barre invisible
                'steps': [
                    {'range': [0, 20], 'color': "#6EE7B7"},
                    {'range': [20, 40], 'color': "#A7F3D0"},
                    {'range': [40, 60], 'color': "#FDE68A"},
                    {'range': [60, 80], 'color': "#FCA5A5"},
                    {'range': [80, 100], 'color': "#EF4444"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': rounded_proba
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)



        st.markdown("---")


        # Section 3 : Affichage du diagnostic du modèle 3 et TOP 3 du modèle 2
        # Titre général
        st.markdown("### 🔍 Diagnostic")

        # Encadré pour la classe prédite
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

        # Top 3
        st.markdown(f"<div style='font-size:16px; white-space: pre-wrap;'>{top_3_text}</div>", unsafe_allow_html=True)


        # Affichage horizontal des images des classes avec proba > 10%
        st.markdown("#### 📸 Exemples des classes détectées (>10%)")

        # Trouver les classes avec proba > 10%
        high_proba_classes = [(idx, proba) for idx, proba in enumerate(result_model2) if proba > 0.10]

        # Créer autant de colonnes que de classes à afficher
        cols = st.columns(len(high_proba_classes))

        # Afficher chaque image dans sa colonne, avec largeur fixe (~1/4 de col2 ≈ 200px)
        for col, (idx, proba) in zip(cols, high_proba_classes):
            class_code = classes[idx].split(' - ')[0]
            class_label = classes[idx]
            image_path = os.path.join("classes", f"{class_code}.jpg")
            if os.path.exists(image_path):
                with col:
                    st.image(image_path, caption=class_label, width=200)

        st.markdown("---")

        # Section 5 : Grad-CAM centrée
        st.markdown("### 🧠 Visualisation Grad-CAM / Zones qui ont impacté l'analyse")
        centered_col = st.columns([1, 2, 1])[1]
        with centered_col:
            st.image(gradcam_image, width=300)
        st.markdown("---")

        # Section 6 : Conseil Skincare
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
            "🍏 Adoptez une alimentation équilibrée riche en antioxydants (fruits, légumes, acides gras essentiels) pour soutenir la santé de votre peau.", unsafe_allow_html=True
        )
