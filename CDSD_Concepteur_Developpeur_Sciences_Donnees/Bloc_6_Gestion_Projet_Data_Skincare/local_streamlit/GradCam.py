# GradCam.py
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
import matplotlib.cm as cm
from keras.models import Model


def _find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

def _normalize_01(x, eps=1e-8):
    x = x - np.min(x)
    x = x / (np.max(x) + eps)
    return x

def generate_gradcam(
    pil_image,
    model,
    preprocess_fn,
    target_size=(224, 224),
    last_conv_layer_name=None,
    class_index=None,
    alpha=0.45,           # intensité de superposition (0-1)
    threshold=0.20,       # masque : on garde les zones > seuil
    colormap_name="jet",  # 'jet', 'turbo', 'magma', 'viridis', ...
    blur=False            # lisser la heatmap (moyenne 3x3)
):
    # 1) Prétraitement
    img_rgb = pil_image.convert("RGB")
    W, H = img_rgb.size
    arr = np.array(img_rgb)
    arr_resized = tf.image.resize(arr, target_size, method="bilinear")
    arr_resized = tf.cast(arr_resized, tf.float32).numpy()
    x = np.expand_dims(arr_resized, axis=0)
    x = preprocess_fn(x)

    # 2) Dernière couche conv
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer_name(model)

    if last_conv_layer_name is not None:
        # ----- Grad-CAM classique -----
        conv_layer = model.get_layer(last_conv_layer_name)

        # Choisit la bonne forme d'inputs sans variable intermédiaire non définie
        in_tensors = model.inputs if isinstance(model.inputs, (list, tuple)) else model.input

        grad_model = Model(
            inputs=in_tensors,
            outputs=[conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            if class_index is None:
                class_index = tf.argmax(preds[0])
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)                # (1,h,w,c)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))       # (c,)
        conv_out = conv_out[0]                               # (h,w,c)

        heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)  # (h,w)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
    else:
        # ----- Fallback: saliency map -----
        x_tf = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            preds = model(x_tf, training=False)
            if class_index is None:
                class_index = tf.argmax(preds[0])
            loss = preds[:, class_index]
        grads = tape.gradient(loss, x_tf)[0].numpy()         # (h,w,3)
        heatmap = np.max(np.abs(grads), axis=-1)             # (h,w)
        heatmap = _normalize_01(heatmap)

    # 3) Optionnel: lissage
    if blur:
        # Moyenne 3x3
        h = heatmap.astype(np.float32)
        h = tf.expand_dims(h, axis=(0, -1))                  # (1,h,w,1)
        h = tf.nn.avg_pool2d(h, ksize=3, strides=1, padding="SAME")
        heatmap = tf.squeeze(h).numpy()

    # 4) Redimensionner à la taille originale
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (H, W)).numpy().squeeze()
    heatmap = _normalize_01(heatmap)

    # 5) Masque de seuil (supprime le “bruit” faible)
    mask = (heatmap >= threshold).astype(np.float32)

    # 6) Colorisation (colormap)
    cmap = cm.get_cmap(colormap_name)
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H,W,3)

    # 7) Superposition RGBA
    base = np.array(img_rgb).astype(np.float32)
    overlay = base * (1.0 - alpha * mask[..., None]) + heatmap_rgb * (alpha * mask[..., None])
    overlay = overlay.clip(0, 255).astype(np.uint8)

    return PILImage.fromarray(overlay)
