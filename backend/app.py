import os, sys, io, base64, random
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import numpy as np

from model_loader import ensure_models
from disease_info  import LABELS, DISEASE_INFO
from leaf_gate     import check_is_plant, compute_suspicion

# ── Boot ────────────────────────────────────────────────────────────────────
print('\n🌿 TariqCViT starting …')
cnn_path, hybrid_path = ensure_models()
DEMO_MODE = (cnn_path is None)

cnn_model    = None
hybrid_model = None

if not DEMO_MODE:
    import tensorflow as tf
    print('Loading CNN model …')
    cnn_model    = tf.keras.models.load_model(cnn_path)
    print('Loading CNN-ViT Hybrid model …')
    hybrid_model = tf.keras.models.load_model(hybrid_path)
    print('✅ Both models ready.\n')

# HSV gate runs in-process — no warmup needed

# ── Flask ────────────────────────────────────────────────────────────────────
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path='')

IMG_SIZE = 224

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def run_inference(image_bytes):
    if DEMO_MODE:
        idx  = random.randint(0, len(LABELS) - 1)
        high = round(random.uniform(0.91, 0.99), 4)
        low  = round(1.0 - high, 4)
        pc = [high, low] if idx == 0 else [low, high]
        ph = [max(0.001, p + random.uniform(-0.03, 0.03)) for p in pc]
        s  = sum(ph); ph = [p/s for p in ph]
        pe = [(a+b)/2 for a, b in zip(pc, ph)]
    else:
        import tensorflow as tf
        arr        = preprocess(image_bytes)
        pc         = cnn_model.predict(arr, verbose=0)[0].tolist()
        hyb_logits = hybrid_model.predict(arr, verbose=0)[0]
        ph         = tf.nn.softmax(hyb_logits).numpy().tolist()
        pe         = [(a+b)/2 for a, b in zip(pc, ph)]

    pred_idx = int(np.argmax(pe))
    label    = LABELS[pred_idx]

    def fmt(probs):
        return {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)}

    # ── Layer 2: entropy / suspicion flag ───────────────────────────────────
    is_suspicious, suspicion_flags, entropy = compute_suspicion(pc, ph, pe)

    return {
        'label':          label,
        'confidence':     round(pe[pred_idx] * 100, 2),
        'probabilities': {
            'cnn':      fmt(pc),
            'hybrid':   fmt(ph),
            'ensemble': fmt(pe),
        },
        'disease_info':    DISEASE_INFO[label],
        'demo_mode':       DEMO_MODE,
        # validation metadata
        'suspicious':      is_suspicious,
        'suspicion_flags': suspicion_flags,
        'entropy':         entropy,
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'demo': DEMO_MODE, 'labels': LABELS})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file        = request.files['image']
    image_bytes = file.read()

    try:
        # ── Layer 1: HSV colour-range plant gate ──────────────────────────────
        is_plant, plant_score, top_labels = check_is_plant(image_bytes)

        if not is_plant:
            return jsonify({
                'rejected':    True,
                'reason':      'No luffa leaf detected in this image.',
                'plant_score': plant_score,
                'top_labels':  top_labels,
                'hint':        'Please upload a clear photo of a Luffa aegyptiaca leaf.',
            }), 200   # 200 so the frontend handles it gracefully, not as an error

        # ── Run disease inference ────────────────────────────────────────────
        result = run_inference(image_bytes)
        result['plant_score'] = plant_score   # expose for frontend badge

        # Encode image for display
        ext  = (file.filename or 'img.jpg').rsplit('.', 1)[-1].lower()
        mime = f'image/{ext}' if ext in ('png', 'gif', 'webp') else 'image/jpeg'
        result['image_b64'] = f'data:{mime};base64,{base64.b64encode(image_bytes).decode()}'

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve React frontend (SPA catch-all)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'🚀 Server on http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
