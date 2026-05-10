"""
04_gradcam_figures.py — Generate Grad-CAM figures for your paper
Saves heatmap overlays for a sample of test images.

Usage:
    python research/04_gradcam_figures.py
    
Output: research/gradcam_outputs/ folder with overlay images
"""
import os, sys
import numpy as np
import tensorflow as tf
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from grad_cam import find_last_conv_layer, compute_gradcam, overlay_heatmap

# ── CONFIG ────────────────────────────────────────────────────────────────────
CNN_MODEL    = 'models/cnn_best.keras'
TEST_DIR     = 'data/test'
OUTPUT_DIR   = 'research/gradcam_outputs'
IMG_SIZE     = 224
N_SAMPLES    = 5   # how many images per class to visualise
# ─────────────────────────────────────────────────────────────────────────────

LABELS = ['Mosaic Disease', 'Insect Infestation']

def preprocess(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE,IMG_SIZE))
    return (np.array(img,np.float32)/255.0)[np.newaxis,...]

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(CNN_MODEL)
    conv_layer = find_last_conv_layer(model)
    print(f'Using conv layer: {conv_layer}')

    for class_idx, class_name in enumerate(LABELS):
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f'Skipping {class_name} — folder not found'); continue
        imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))][:N_SAMPLES]

        for fname in imgs:
            fpath = os.path.join(class_dir, fname)
            arr   = preprocess(fpath)
            heatmap = compute_gradcam(model, arr, class_idx, conv_layer)

            with open(fpath,'rb') as f:
                orig_bytes = f.read()
            b64 = overlay_heatmap(orig_bytes, heatmap, alpha=0.5)

            # Decode b64 and save
            import base64
            out_bytes = base64.b64decode(b64)
            out_path  = os.path.join(OUTPUT_DIR, f'{class_name.replace(" ","_")}_{fname}')
            with open(out_path,'wb') as f:
                f.write(out_bytes)
            print(f'Saved: {out_path}')

    print(f'\nDone. Check {OUTPUT_DIR}/')
