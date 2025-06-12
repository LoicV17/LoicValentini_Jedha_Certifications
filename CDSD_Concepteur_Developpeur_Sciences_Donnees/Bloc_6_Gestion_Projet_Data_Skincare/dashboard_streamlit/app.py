import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
import plotly.graph_objects as go
from GradCam import generate_gradcam

# --- Chargement des modèles ---
model1 = tf.keras.models.load_model("models/model1_skindisease.h5")
model2 = tf.keras.models.load_model("models/model2_ham10000.h5")
model3 = joblib.load("models/model3_stacking.joblib")

# Classes du modèle 2 (HAM10000)
classes = {
    0: 'akiec - kératoses actiniques',
    1: 'bcc - carcinome basocellulaire',
    2: 'bkl - kératoses séborrhéiques',
    3: 'df - dermatofibromes',
    4: 'mel - mélanome',
    5: 'nv - nævus mélanocytaire',
    6: 'vasc - lésions vasculaires'
}

# --- Prétraitement ---
def preprocess_image_model1(pil_img):
    img = pil_img.resize((240, 240))
    img_array = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_model2(pil_img):
    img = pil_img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_effnet(np.expand_dims(img_array, axis=0))
    return img_array

# --- Prédiction ---
def predict_combined(pil_image, age, sex, localization):
    # Model 1: bénin / malin
    img1 = preprocess_image_model1(pil_image)
    proba_malin_model1 = model1.predict(img1, verbose=0)[0][0]

    # Model 2: classification HAM10000
    img2 = preprocess_image_model2(pil_image)
    pred_model2 = model2.predict(img2, verbose=0)[0]

    # GradCAM sur modèle 2
    gradcam_img = generate_gradcam(pil_image.resize((224, 224)))

    # Top 3 classes
    top_indices = np.argsort(pred_model2)[::-1][:3]
    top3_text = "\n".join([f"{classes[i]} : {pred_model2[i]*100:.1f}%" for i in top_indices])

    # Input for model 3
    input_data = {
        "age": age,
        "sex": sex,
        "localization": localization,
        "proba_akiec": pred_model2[0],
        "proba_bcc": pred_model2[1],
        "proba_bkl": pred_model2[2],
        "proba_df": pred_model2[3],
        "proba_mel": pred_model2[4],
        "proba_nv": pred_model2[5],
        "proba_vasc": pred_model2[6],
    }
    df_input = pd.DataFrame([input_data])
    prediction_model3 = model3.predict(df_input)[0]

    return gradcam_img, proba_malin_model1, pred_model2, top3_text, prediction_model3

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🔎 Skin Care - Analyse des grains de beauté 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Soumettez une image et obtenez une prédiction du caractère bénin/malin, ainsi qu'une classification dermatologique.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

image = None

with col1:
    st.subheader("📥 Import manuel ou Webcam")
    camera_image = st.camera_input("Prenez une photo")
    uploaded_file = st.file_uploader("Ou choisissez une image JPG...", type="jpg")

    st.markdown("---")
    st.subheader("📁 Ou utilisez un exemple")
    example_files = ["Exemple1.jpg", "Exemple2.jpg", "Exemple3.jpg"]
    selected_example = st.selectbox("Choisissez un exemple :", ["-- Aucun --"] + example_files)

    st.subheader("👤 Informations Patient")
    age = st.slider("Âge", 0, 100, 30)
    sex = st.selectbox("Sexe", ["male", "female"])
    localization = st.selectbox("Localisation", [
        "scalp", "ear", "face", "back", "chest", "trunk",
        "upper extremity", "lower extremity", "genital", "abdomen", "unknown"
    ])

    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Photo capturée via la webcam", use_column_width=True)
    elif uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée", use_column_width=True)
    elif selected_example != "-- Aucun --":
        example_path = os.path.join("examples", selected_example)
        image = Image.open(example_path)
        st.image(image, caption=f"Exemple : {selected_example}", use_column_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

with col2:
    if image:
        gradcam_img, proba_m1, probs_m2, top3, pred_m3 = predict_combined(image, age, sex, localization)
        combined_score = (proba_m1 + probs_m2[4]) / 2 * 100

        st.markdown("### 🧪 Jauge bénin / malin (combinée)")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=combined_score,
            number={'suffix': "%"},
            title={'text': "Score de risque combiné"},
            gauge={
                'axis': {'range': [0, 100]},
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
                    'value': combined_score
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 GradCAM")
        st.image(gradcam_img, width=300)

        st.markdown("### 🔍 Top 3 classes (model2)")
        st.markdown(f"<pre>{top3}</pre>", unsafe_allow_html=True)

        st.markdown("### 🧾 Prédiction finale (stacking model)")
        st.success(f"Classe prédite : {pred_m3}")

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

st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
