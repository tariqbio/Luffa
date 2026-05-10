import os, sys, io, base64, random
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import numpy as np

from model_loader import ensure_models
from disease_info  import LABELS, DISEASE_INFO
from leaf_gate     import check_is_plant, compute_suspicion
from grad_cam      import find_last_conv_layer, compute_gradcam, overlay_heatmap

# ── Boot ──────────────────────────────────────────────────────────────────────
print('\n🌿 TariqCViT starting …')
cnn_path, hybrid_path = ensure_models()
DEMO_MODE = (cnn_path is None)

cnn_model = hybrid_model = None
CNN_CONV_LAYER = HYBRID_CONV_LAYER = None

if not DEMO_MODE:
    import tensorflow as tf
    print('Loading CNN model …')
    cnn_model    = tf.keras.models.load_model(cnn_path)
    print('Loading CNN-ViT Hybrid model …')
    hybrid_model = tf.keras.models.load_model(hybrid_path)
    CNN_CONV_LAYER    = find_last_conv_layer(cnn_model)
    HYBRID_CONV_LAYER = find_last_conv_layer(hybrid_model)
    print(f'✅ Models ready. CNN conv layer: {CNN_CONV_LAYER}  Hybrid conv layer: {HYBRID_CONV_LAYER}\n')

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path='')
IMG_SIZE = 224


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return (np.array(img, np.float32) / 255.0)[np.newaxis, ...]


def make_thumbnail_b64(image_bytes: bytes, size: int = 120) -> str:
    """Small thumbnail for history panel."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def encode_b64(image_bytes: bytes, filename: str) -> str:
    ext  = (filename or 'img.jpg').rsplit('.', 1)[-1].lower()
    mime = f'image/{ext}' if ext in ('png','gif','webp') else 'image/jpeg'
    return f'data:{mime};base64,{base64.b64encode(image_bytes).decode()}'


def run_inference(image_bytes: bytes, want_gradcam: bool = True) -> dict:
    if DEMO_MODE:
        idx  = random.randint(0, len(LABELS) - 1)
        high = round(random.uniform(0.91, 0.99), 4)
        low  = round(1.0 - high, 4)
        pc = [high, low] if idx == 0 else [low, high]
        ph = [max(0.001, p + random.uniform(-0.03, 0.03)) for p in pc]
        s  = sum(ph); ph = [p/s for p in ph]
        pe = [(a+b)/2 for a,b in zip(pc, ph)]
        gradcam_b64 = None
    else:
        import tensorflow as tf
        arr  = preprocess(image_bytes)
        pc   = cnn_model.predict(arr, verbose=0)[0].tolist()
        ph   = tf.nn.softmax(hybrid_model.predict(arr, verbose=0)[0]).numpy().tolist()
        pe   = [(a+b)/2 for a,b in zip(pc, ph)]

        gradcam_b64 = None
        if want_gradcam and CNN_CONV_LAYER:
            try:
                pred_idx = int(np.argmax(pe))
                heatmap  = compute_gradcam(cnn_model, arr, pred_idx, CNN_CONV_LAYER)
                gradcam_b64 = overlay_heatmap(image_bytes, heatmap)
            except Exception as e:
                print(f'Grad-CAM skipped: {e}')

    pred_idx = int(np.argmax(pe))
    label    = LABELS[pred_idx]

    def fmt(p): return {LABELS[i]: round(v*100,2) for i,v in enumerate(p)}

    is_suspicious, suspicion_flags, entropy = compute_suspicion(pc, ph, pe)

    return {
        'label':           label,
        'confidence':      round(pe[pred_idx]*100, 2),
        'probabilities':   {'cnn': fmt(pc), 'hybrid': fmt(ph), 'ensemble': fmt(pe)},
        'disease_info':    DISEASE_INFO[label],
        'demo_mode':       DEMO_MODE,
        'suspicious':      is_suspicious,
        'suspicion_flags': suspicion_flags,
        'entropy':         entropy,
        'gradcam_b64':     gradcam_b64,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status':'ok','demo':DEMO_MODE,'labels':LABELS})


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'No image file provided'}), 400
    file        = request.files['image']
    image_bytes = file.read()
    try:
        is_plant, plant_score = check_is_plant(image_bytes)
        if not is_plant:
            return jsonify({'rejected':True,'reason':'No luffa leaf detected.',
                            'plant_score':plant_score,
                            'hint':'Upload a clear photo of a Luffa aegyptiaca leaf.'}), 200

        result = run_inference(image_bytes, want_gradcam=True)
        result['plant_score'] = plant_score
        result['image_b64']   = encode_b64(image_bytes, file.filename or 'img.jpg')
        result['thumb_b64']   = make_thumbnail_b64(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error':'No images provided'}), 400
    results = []
    for f in files[:10]:   # cap at 10 per batch
        image_bytes = f.read()
        try:
            is_plant, plant_score = check_is_plant(image_bytes)
            if not is_plant:
                results.append({'filename':f.filename,'rejected':True,
                                 'reason':'No luffa leaf detected.',
                                 'plant_score':plant_score,
                                 'image_b64':encode_b64(image_bytes, f.filename)})
                continue
            r = run_inference(image_bytes, want_gradcam=False)  # skip gradcam in batch for speed
            r['plant_score'] = plant_score
            r['image_b64']   = encode_b64(image_bytes, f.filename)
            r['thumb_b64']   = make_thumbnail_b64(image_bytes)
            r['filename']    = f.filename
            results.append(r)
        except Exception as e:
            results.append({'filename':f.filename,'error':str(e)})
    return jsonify({'results': results, 'count': len(results)})


@app.route('/', defaults={'path':''})
@app.route('/<path:path>')
def serve_frontend(path):
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'🚀 TariqCViT on http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
