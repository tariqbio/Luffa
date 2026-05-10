"""
01_metrics.py — Full evaluation metrics for your trained models
Run this locally with your models and dataset.

Usage:
    python research/01_metrics.py

Edit the paths below before running.
"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── CONFIG — edit these ───────────────────────────────────────────────────────
CNN_MODEL    = 'models/cnn_best.keras'
HYBRID_MODEL = 'models/hybrid_best.keras'
TEST_DIR     = 'data/test'          # folder with subfolders per class
IMG_SIZE     = 224
BATCH_SIZE   = 32
CLASS_NAMES  = ['Mosaic Disease', 'Insect Infestation']  # must match subfolder names
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, label_mode='int'
    )
    # Normalise
    norm = tf.keras.layers.Rescaling(1./255)
    return ds.map(lambda x,y: (norm(x),y)), ds.class_names

def evaluate_model(model, ds, name):
    print(f'\n── {name} ──')
    all_true, all_pred, all_probs = [], [], []
    for imgs, labels in ds:
        logits = model.predict(imgs, verbose=0)
        probs  = tf.nn.softmax(logits).numpy() if logits.shape[-1] > 1 else logits.numpy()
        preds  = np.argmax(probs, axis=1)
        all_true.extend(labels.numpy()); all_pred.extend(preds); all_probs.extend(probs)
    y_true  = np.array(all_true)
    y_pred  = np.array(all_pred)
    y_probs = np.array(all_probs)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, cohen_kappa_score)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    auc = roc_auc_score(y_true, y_probs[:,1])
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f'AUC-ROC: {auc:.4f}  |  Cohen Kappa: {kappa:.4f}')
    return y_true, y_pred, y_probs

if __name__ == '__main__':
    print('Loading models…')
    cnn    = tf.keras.models.load_model(CNN_MODEL)
    hybrid = tf.keras.models.load_model(HYBRID_MODEL)
    ds, cls = load_test_data()
    print(f'Classes detected: {cls}')
    yt_c,  yp_c,  yprob_c  = evaluate_model(cnn,    ds, 'CNN Baseline')
    yt_h,  yp_h,  yprob_h  = evaluate_model(hybrid, ds, 'CNN-ViT Hybrid')
    # Ensemble
    ens_probs = (yprob_c + yprob_h) / 2
    ens_preds = np.argmax(ens_probs, axis=1)
    from sklearn.metrics import classification_report, roc_auc_score
    print('\n── Ensemble (Soft Voting) ──')
    print(classification_report(yt_c, ens_preds, target_names=CLASS_NAMES, digits=4))
    print(f'Ensemble AUC-ROC: {roc_auc_score(yt_c, ens_probs[:,1]):.4f}')
