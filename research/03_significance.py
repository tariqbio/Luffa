"""
03_significance.py — Statistical significance testing
Tests whether CNN-ViT Hybrid significantly outperforms CNN baseline.

Uses McNemar's test (correct test for comparing two classifiers on same test set).

Usage:
    python research/03_significance.py
"""
import numpy as np
import tensorflow as tf
from scipy.stats import chi2

# ── CONFIG ────────────────────────────────────────────────────────────────────
CNN_MODEL    = 'models/cnn_best.keras'
HYBRID_MODEL = 'models/hybrid_best.keras'
TEST_DIR     = 'data/test'
IMG_SIZE     = 224
BATCH_SIZE   = 32
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, ds):
    preds, labels = [], []
    for imgs, ys in ds:
        logits = model.predict(imgs, verbose=0)
        probs  = tf.nn.softmax(logits).numpy()
        preds.extend(np.argmax(probs, axis=1))
        labels.extend(ys.numpy())
    return np.array(preds), np.array(labels)

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test.
    H0: both models make the same errors (no significant difference)
    If p < 0.05 → the difference is statistically significant
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right
    # With continuity correction (Yates)
    chi2_stat = ((abs(b-c)-1)**2) / (b+c) if (b+c)>0 else 0
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value, b, c

if __name__ == '__main__':
    print('Loading models and test data…')
    norm = tf.keras.layers.Rescaling(1./255)
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)
    ds = ds.map(lambda x,y:(norm(x),y))

    cnn    = tf.keras.models.load_model(CNN_MODEL)
    hybrid = tf.keras.models.load_model(HYBRID_MODEL)
    cnn_pred,  y_true = get_predictions(cnn,    ds)
    hyb_pred,  _      = get_predictions(hybrid, ds)

    print(f'\nCNN accuracy:           {np.mean(cnn_pred==y_true)*100:.4f}%')
    print(f'CNN-ViT Hybrid accuracy: {np.mean(hyb_pred==y_true)*100:.4f}%')

    chi2_stat, p, b, c = mcnemar_test(y_true, cnn_pred, hyb_pred)
    print(f'\nMcNemar\'s Test (with Yates correction)')
    print(f'  CNN correct, Hybrid wrong: {b}')
    print(f'  CNN wrong,   Hybrid correct: {c}')
    print(f'  χ² = {chi2_stat:.4f}')
    print(f'  p-value = {p:.6f}')
    if p < 0.05:
        print(f'\n✅ SIGNIFICANT (p={p:.4f} < 0.05)')
        print('  The CNN-ViT Hybrid significantly outperforms the CNN baseline.')
    else:
        print(f'\n⚠  NOT SIGNIFICANT (p={p:.4f} ≥ 0.05)')
        print('  The difference may be due to random variation.')

    # Also report 95% CI on accuracy difference
    n = len(y_true)
    diff = np.mean(hyb_pred==y_true) - np.mean(cnn_pred==y_true)
    se   = np.sqrt((diff*(1-diff))/n)
    print(f'\nAccuracy difference: {diff*100:.4f}% (95% CI: ±{1.96*se*100:.4f}%)')
