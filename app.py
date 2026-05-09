"""
LuffaGuard — Leaf Disease Detection Backend
============================================
Loads your two trained .keras models and serves predictions via a REST API.

Usage:
    python app.py

Then open: http://localhost:5000
"""

import os, io, base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────────
# Point these to wherever you saved cnn_best.keras and hybrid_best.keras
CNN_MODEL_PATH    = os.environ.get("CNN_MODEL",    "models/cnn_best.keras")
HYBRID_MODEL_PATH = os.environ.get("HYBRID_MODEL", "models/hybrid_best.keras")

IMG_SIZE    = 224          # both models use 224×224
LABELS      = ["Mosaic Disease", "Insect Infestation"]
DEMO_MODE   = not (os.path.exists(CNN_MODEL_PATH) and os.path.exists(HYBRID_MODEL_PATH))

# ── Disease knowledge base ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "Mosaic Disease": {
        "scientific_name": "Cucumber Mosaic Virus (CMV)",
        "severity_levels": ["Mild", "Moderate", "Severe"],
        "overview": (
            "Mosaic Disease in Luffa aegyptiaca is primarily caused by Cucumber Mosaic Virus (CMV), "
            "transmitted by aphids (Myzus persicae) and through infected seed. It causes characteristic "
            "mosaic patterns of alternating light and dark green patches on the leaves, leading to "
            "chlorosis, leaf curling, distortion, and stunted fruit development."
        ),
        "symptoms": [
            "Yellow-green mosaic patterning on leaf lamina",
            "Leaf blade distortion, puckering, and curling downward",
            "Stunted plant growth and reduced internode length",
            "Mottled, deformed fruits with reduced market quality",
            "Yellowing along leaf veins (vein clearing in early stages)",
            "Necrotic lesions in advanced infection stages",
        ],
        "causes": [
            "Aphid vectors (Myzus persicae, Aphis gossypii) — primary transmission",
            "Infected seed or planting material",
            "Manual transmission via contaminated tools or hands",
            "Infected weeds or neighbouring cucurbit crops acting as reservoirs",
        ],
        "treatment": [
            "Remove and destroy infected plants immediately to prevent spread",
            "Apply systemic insecticides (Imidacloprid 17.8% SL @ 0.3 ml/L) to control aphid vectors",
            "Use mineral oil sprays (petroleum oil 2%) to reduce aphid probing and virus acquisition",
            "Foliar spray of micronutrient mixtures (Zinc 0.5% + Boron 0.1%) to support immune response",
            "No direct chemical cure exists — focus on vector control and resistance",
            "Apply Trichoderma-based bioagents to strengthen root health",
        ],
        "prevention": [
            "Use certified virus-free seeds from reputable suppliers",
            "Grow virus-tolerant or resistant varieties where available",
            "Install yellow sticky traps to monitor and reduce aphid population",
            "Maintain strict field hygiene — sanitise tools with 1% sodium hypochlorite",
            "Intercrop with maize or sorghum as barrier crops to reduce aphid flight",
            "Remove weeds (especially cucurbit family weeds) that serve as CMV reservoirs",
            "Avoid working in the field when foliage is wet to reduce mechanical spread",
        ],
        "economic_impact": (
            "Yield losses of 30–80% reported in severe outbreaks. "
            "Mosaic-infected fruits are unmarketable, causing significant income loss "
            "for smallholder farmers in South and Southeast Asia."
        ),
        "color": "#e67e22",
        "icon": "🦠",
        "urgency": "HIGH — Act within 24–48 hours",
    },
    "Insect Infestation": {
        "scientific_name": "Multiple species — Epilachna, Aulacophora, Thrips, Aphids",
        "severity_levels": ["Low", "Moderate", "High"],
        "overview": (
            "Insect infestation in Luffa aegyptiaca involves multiple pest species including "
            "epilachna beetles (Henosepilachna vigintioctopunctata), red pumpkin beetle "
            "(Aulacophora foveicollis), thrips (Thrips palmi), and various aphid species. "
            "Damage manifests as irregular leaf skeletonisation, shot-holes, scraping marks, "
            "silvery streaks, and general defoliation that weakens the plant."
        ),
        "symptoms": [
            "Irregular holes, shot-holes, and skeletonised patches on leaf surface",
            "Scraping damage leaving thin papery epidermis (epilachna characteristic)",
            "Silvery streaks or bronzing caused by thrips feeding",
            "Curled, distorted new growth from aphid colonies",
            "Presence of frass (insect excrement) on leaf undersides",
            "Webbing on leaf undersides indicating spider mite presence",
            "Wilting of young vines due to root/stem feeding",
        ],
        "causes": [
            "Epilachna beetles — adult and larval scraping of leaf tissue",
            "Red pumpkin beetles — feeding on cotyledons, young leaves, and flowers",
            "Thrips (Thrips palmi) — rasping-sucking damage causing silvery streaks",
            "Aphids — colony feeding on young growth, excreting honeydew",
            "Whiteflies — phloem feeding and honeydew/sooty mould complex",
            "Spider mites — population explosions in dry conditions",
        ],
        "treatment": [
            "Spray Neem oil extract (5 ml/L) or NSKE 5% — broad-spectrum biopesticide",
            "Apply Spinosad 45 SC (0.3 ml/L) for thrips and beetles — low mammalian toxicity",
            "Use Imidacloprid 17.8% SL (0.3 ml/L) for aphid and whitefly control",
            "Profenophos + Cypermethrin (2 ml/L) for severe mixed infestations",
            "Introduce Trichogramma parasitoids for biological control of eggs",
            "Manual removal of egg masses and early instar larvae from undersides of leaves",
            "Spray 3–4 times at 7–10 day intervals until infestation is suppressed",
        ],
        "prevention": [
            "Install insect-proof netting on nursery beds to protect seedlings",
            "Use yellow and blue sticky traps (5–10 per 1000 m²) for thrips and whitefly",
            "Practice crop rotation — avoid planting cucurbits in the same field consecutively",
            "Intercrop with aromatic plants (basil, coriander) to repel pest insects",
            "Maintain field sanitation — remove crop debris promptly after harvest",
            "Monitor weekly for early detection, especially leaf undersides",
            "Conserve natural enemies (ladybird beetles, lacewings, parasitic wasps)",
        ],
        "economic_impact": (
            "Yield losses of 20–60% depending on infestation severity. "
            "Defoliation weakens plants, reducing photosynthetic capacity and fruit set. "
            "Insect-vectored secondary infections can compound losses significantly."
        ),
        "color": "#27ae60",
        "icon": "🐛",
        "urgency": "MODERATE — Monitor and treat within 3–5 days",
    },
    "Healthy": {
        "scientific_name": "No pathogen detected",
        "severity_levels": ["None"],
        "overview": (
            "The leaf appears healthy with no visible disease or significant insect damage. "
            "Continue regular monitoring and preventive care to maintain plant health."
        ),
        "symptoms": ["No disease symptoms detected"],
        "causes": [],
        "treatment": ["Continue normal agricultural practices"],
        "prevention": [
            "Maintain regular field monitoring (twice weekly)",
            "Keep irrigation schedules consistent",
            "Apply balanced fertilisation (NPK + micronutrients)",
            "Remove any weeds that may harbour pests",
        ],
        "economic_impact": "No current threat detected.",
        "color": "#2ecc71",
        "icon": "✅",
        "urgency": "LOW — Routine monitoring only",
    },
}

# ── Model loading ──────────────────────────────────────────────────────────────
app = Flask(__name__)
cnn_model    = None
hybrid_model = None

def load_models():
    global cnn_model, hybrid_model
    if DEMO_MODE:
        print("⚠️  Model files not found — running in DEMO mode (random predictions).")
        print(f"   Expected CNN    : {CNN_MODEL_PATH}")
        print(f"   Expected Hybrid : {HYBRID_MODEL_PATH}")
        return
    print("Loading CNN model …")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("Loading CNN-ViT Hybrid model …")
    hybrid_model = tf.keras.models.load_model(HYBRID_MODEL_PATH)
    print("✅ Both models loaded.")


def preprocess_image(image_bytes):
    """Match the notebook preprocessing exactly: resize to 224×224, divide by 255."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # shape: (1, 224, 224, 3)


def predict(image_bytes):
    """Returns dict with CNN, Hybrid, and Ensemble predictions."""
    arr = preprocess_image(image_bytes)

    if DEMO_MODE:
        # Simulate realistic-looking predictions for demo
        import random
        label_idx = random.randint(0, len(LABELS) - 1)
        high = round(random.uniform(0.92, 0.99), 4)
        low  = round(1.0 - high, 4)
        probs_cnn    = [high, low] if label_idx == 0 else [low, high]
        probs_hybrid = [max(0, p + random.uniform(-0.03, 0.03)) for p in probs_cnn]
        s = sum(probs_hybrid); probs_hybrid = [p/s for p in probs_hybrid]
        probs_ens = [(a+b)/2 for a, b in zip(probs_cnn, probs_hybrid)]
    else:
        cnn_probs    = cnn_model.predict(arr, verbose=0)[0].tolist()
        hyb_logits   = hybrid_model.predict(arr, verbose=0)[0]
        hyb_probs    = tf.nn.softmax(hyb_logits).numpy().tolist()
        probs_cnn    = cnn_probs
        probs_hybrid = hyb_probs
        probs_ens    = [(a+b)/2 for a, b in zip(probs_cnn, probs_hybrid)]

    pred_idx = int(np.argmax(probs_ens))
    label    = LABELS[pred_idx]

    return {
        "label":          label,
        "label_index":    pred_idx,
        "confidence":     round(probs_ens[pred_idx] * 100, 2),
        "probabilities": {
            "cnn":    {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs_cnn)},
            "hybrid": {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs_hybrid)},
            "ensemble": {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs_ens)},
        },
        "disease_info":   DISEASE_INFO.get(label, DISEASE_INFO["Healthy"]),
        "demo_mode":      DEMO_MODE,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file  = request.files["image"]
    image_bytes = file.read()
    try:
        result = predict(image_bytes)

        # Encode the uploaded image as base64 so frontend can show it
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        ext     = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpeg"
        mime    = f"image/{ext}" if ext in ("png","gif","webp") else "image/jpeg"
        result["image_b64"]  = f"data:{mime};base64,{b64_img}"

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_models()
    print("\n🌿 LuffaGuard is running → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
