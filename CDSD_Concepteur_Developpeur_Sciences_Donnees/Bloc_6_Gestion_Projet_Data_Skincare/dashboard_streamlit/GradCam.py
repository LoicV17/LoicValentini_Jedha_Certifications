import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from matplotlib import cm

# Charger le modèle pré-entrainé

def generate_gradcam(pil_img):
    from tensorflow.keras.models import Model
    import matplotlib.cm as cm

    # Préparation image
    img = pil_img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_effnet(np.expand_dims(img_array, axis=0))

    # Chargement modèle 2
    model = tf.keras.models.load_model("models/model2_ham10000.h5")

    # Dernière couche convolutive à capturer
    last_conv_layer = model.get_layer("top_conv")  # ou essaie "block7a_project_bn" si erreur
    last_conv_model = Model(model.inputs, last_conv_layer.output)

    # Classifieur à partir de cette couche
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[-2:]:
        x = layer(x)
    classifier_model = Model(classifier_input, x)

    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_output = last_conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        top_class = tf.argmax(preds[0])
        top_output = preds[:, top_class]

    grads = tape.gradient(top_output, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_output = conv_output[0].numpy()
    heatmap = np.dot(conv_output, pooled_grads.numpy())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Application de la colormap
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * 0.4 + keras_image.img_to_array(pil_img.resize((224, 224)))
    superimposed = keras_image.array_to_img(superimposed)

    return superimposed
