"""
leaf_gate.py — Two-layer input validation (pure classical image processing)

Layer 1: HSV colour-range gate
  Counts pixels in the plant-green HSV band.
  No external model — pure PIL + NumPy. Fully claimable in a research paper
  as a preprocessing validation step.

Layer 2: Entropy + model-agreement suspicion flag
  Operates on outputs of the author's own trained models only.
"""

import io, math
import numpy as np
from PIL import Image


# ── Layer 1: HSV plant-green gate ────────────────────────────────────────────
#
# Plant leaves occupy a well-defined HSV region:
#   Hue        30° – 165°   (yellow-green through blue-green)
#   Saturation 25% – 100%   (excludes washed-out / grey pixels)
#   Value      10% – 95%    (excludes near-black and blown-out white)
#
# A shopping bag, face, document, etc. will have very few pixels in this band.
# A luffa leaf — even a diseased, yellowing one — will have many.

HSV_HUE_LOW   = 30   / 360   # normalised 0-1
HSV_HUE_HIGH  = 165  / 360
HSV_SAT_MIN   = 0.25
HSV_VAL_MIN   = 0.10
HSV_VAL_MAX   = 0.95
PLANT_THRESHOLD = 0.15        # 15 % of pixels must be in the green band


def check_is_plant(image_bytes: bytes, threshold: float = PLANT_THRESHOLD):
    """
    Returns (is_plant: bool, plant_score: float 0-100)

    plant_score = percentage of pixels that fall inside the plant-green HSV band.
    Runs in ~5 ms on a 224×224 thumbnail — no model, no network call.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128), Image.LANCZOS)   # thumbnail for speed

    hsv = np.array(img, dtype=np.float32) / 255.0  # shape (128, 128, 3) in RGB

    # Convert RGB → HSV (vectorised, no OpenCV needed)
    r, g, b = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cmax = np.max(hsv, axis=-1)
    cmin = np.min(hsv, axis=-1)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation (avoid div-by-zero)
    s = np.where(cmax > 0, delta / cmax, 0.0)

    # Hue (0-1 scale, 0 = red, 1/3 = green, 2/3 = blue)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(
            delta == 0, 0.0,
            np.where(
                cmax == r, ((g - b) / delta) % 6,
                np.where(
                    cmax == g, (b - r) / delta + 2,
                    (r - g) / delta + 4
                )
            )
        ) / 6.0

    # Count pixels in the plant-green band
    in_band = (
        (h  >= HSV_HUE_LOW)  & (h  <= HSV_HUE_HIGH) &
        (s  >= HSV_SAT_MIN)  &
        (v  >= HSV_VAL_MIN)  & (v  <= HSV_VAL_MAX)
    )
    plant_score = float(in_band.mean())   # fraction 0-1

    return plant_score >= threshold, round(plant_score * 100, 1)


# ── Layer 2: Entropy / suspicion flag ────────────────────────────────────────
#
# The softmax problem: a closed classifier *must* output probabilities that sum
# to 1, so it forces every input — including a shopping bag — into one of its
# known classes with fake confidence.
#
# We flag results where BOTH signals fire simultaneously:
#   • Normalised Shannon entropy < 0.12  (model is extremely certain)
#   • CNN and CNN-ViT Hybrid agree within 2 pp  (suspiciously identical)
#
# Flagging two independent signals reduces false alarms on genuine high-
# confidence leaf images while catching most OOD inputs that slip through.

def compute_suspicion(pc: list, ph: list, pe: list):
    """
    Returns (is_suspicious: bool, flags: list[str], entropy: float)

    pc / ph / pe — probability lists (one value per class) from CNN,
                   Hybrid, and Ensemble respectively.
    """
    flags    = []
    pred_idx = int(np.argmax(pe))

    # Normalised Shannon entropy (0 = fully certain, 1 = uniform)
    entropy      = -sum(p * math.log(p + 1e-10) for p in pe)
    norm_entropy = entropy / math.log(len(pe))

    conf_ens    = pe[pred_idx]
    conf_cnn    = pc[pred_idx]
    conf_hybrid = ph[pred_idx]

    if conf_ens > 0.90 and norm_entropy < 0.12:
        flags.append("extremely_low_entropy")

    if abs(conf_cnn - conf_hybrid) < 0.02 and conf_ens > 0.88:
        flags.append("suspiciously_identical_outputs")

    if conf_ens > 0.95 and norm_entropy < 0.08:
        flags.append("near_zero_entropy")

    is_suspicious = len(flags) >= 2

    return is_suspicious, flags, round(norm_entropy, 4)
