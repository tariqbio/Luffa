"""
grad_cam.py — Gradient-weighted Class Activation Mapping
Pure TensorFlow + NumPy + PIL. No OpenCV, no matplotlib.
Academically owned: operates solely on the author's trained model outputs.
"""
import io, base64
import numpy as np
from PIL import Image
import tensorflow as tf


def find_last_conv_layer(model) -> str:
    """Auto-detect the last Conv2D layer in any Keras model or submodel."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):  # nested submodel
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found in model.")


def compute_gradcam(model, img_array: np.ndarray, class_idx: int, conv_layer_name: str) -> np.ndarray:
    """
    Returns (H_feat, W_feat) heatmap, values in [0, 1].
    img_array: (1, 224, 224, 3), float32, [0,1].
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )
    img_t = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        conv_out, preds = grad_model(img_t, training=False)
        probs = tf.nn.softmax(preds)
        score = probs[:, class_idx]

    grads       = tape.gradient(score, conv_out)          # (1,h,w,C)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    heatmap     = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])  # (h,w)
    heatmap     = tf.nn.relu(heatmap)
    mx          = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap.numpy()


def _jet(h: np.ndarray) -> np.ndarray:
    """Jet colormap without matplotlib. h in [0,1] → (H,W,3) uint8."""
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(original_bytes: bytes, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    """
    Blend Grad-CAM heatmap onto the original image.
    Returns base64-encoded JPEG string.
    """
    orig     = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    h_img    = Image.fromarray((heatmap * 255).astype(np.uint8), 'L').resize(orig.size, Image.LANCZOS)
    h_arr    = np.array(h_img, np.float32) / 255.0
    jet_img  = Image.fromarray(_jet(h_arr), 'RGB')

    orig_f   = np.array(orig, np.float32)
    jet_f    = np.array(jet_img, np.float32)
    w        = h_arr[..., np.newaxis]
    blended  = np.clip(orig_f * (1 - alpha * w) + jet_f * (alpha * w), 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, 'JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode()
