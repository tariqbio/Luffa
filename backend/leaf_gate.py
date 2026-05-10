"""
leaf_gate.py — Two-layer input validation (pure classical image processing)

Layer 1: HSV colour-range gate  +  spatial coverage check
  Counts green pixels AND verifies they are spatially spread across the frame.
  A leaf covers most of the image. A green pen covers only a thin strip.

Layer 2: Entropy + model-agreement suspicion flag
  Operates on outputs of the author's own trained models only.

Both layers are academically claimable as preprocessing steps —
no external model, no third-party weights.
"""

import io, math
import numpy as np
from PIL import Image


# ── Layer 1: HSV plant-green gate ────────────────────────────────────────────
HSV_HUE_LOW     = 30  / 360    # yellow-green
HSV_HUE_HIGH    = 165 / 360    # blue-green
HSV_SAT_MIN     = 0.25
HSV_VAL_MIN     = 0.10
HSV_VAL_MAX     = 0.95

PLANT_THRESHOLD   = 0.22   # raised from 0.15 → rejects 18-21% green objects (pens, bags)
COVERAGE_GRID     = 4      # divide image into 4×4 = 16 regions
COVERAGE_MIN_FRAC = 0.10   # each qualifying region must have ≥10 % green pixels
COVERAGE_MIN_REGS = 5      # at least 5 of 16 regions must qualify → rejects thin objects


def _rgb_to_hsv(arr: np.ndarray):
    """Vectorised RGB→HSV. arr: (H,W,3) float32 in [0,1]. Returns h,s,v each (H,W)."""
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    cmax  = np.max(arr, axis=-1)
    cmin  = np.min(arr, axis=-1)
    delta = cmax - cmin
    v = cmax
    s = np.where(cmax > 0, delta / cmax, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        h = np.where(delta == 0, 0.0,
            np.where(cmax == r, ((g - b) / delta) % 6,
            np.where(cmax == g,  (b - r) / delta + 2,
                                 (r - g) / delta + 4))) / 6.0
    return h, s, v


def check_is_plant(image_bytes: bytes,
                   threshold: float = PLANT_THRESHOLD):
    """
    Returns (is_plant: bool, plant_score: float 0–100).

    Two-signal check:
      1. Global green pixel ratio ≥ threshold
      2. Green pixels spread across ≥ COVERAGE_MIN_REGS grid regions
         (rejects thin green objects like pens, rulers, bags with green strip)

    Runs in ~6 ms on a 128×128 thumbnail. No model, no network call.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128), Image.LANCZOS)
    arr = np.array(img, np.float32) / 255.0   # (128, 128, 3)

    h, s, v = _rgb_to_hsv(arr)

    in_band = (
        (h >= HSV_HUE_LOW) & (h <= HSV_HUE_HIGH) &
        (s >= HSV_SAT_MIN) &
        (v >= HSV_VAL_MIN) & (v <= HSV_VAL_MAX)
    )

    plant_score = float(in_band.mean())   # global fraction

    # ── Spatial coverage check ──────────────────────────────────────────────
    # Split into COVERAGE_GRID × COVERAGE_GRID regions, count qualifying ones.
    G   = COVERAGE_GRID
    sz  = 128 // G  # region size in pixels
    qualifying_regions = 0
    for row in range(G):
        for col in range(G):
            region = in_band[row*sz:(row+1)*sz, col*sz:(col+1)*sz]
            if region.mean() >= COVERAGE_MIN_FRAC:
                qualifying_regions += 1

    spatially_distributed = qualifying_regions >= COVERAGE_MIN_REGS

    is_plant = (plant_score >= threshold) and spatially_distributed

    return is_plant, round(plant_score * 100, 1)


# ── Layer 2: Entropy / suspicion flag ────────────────────────────────────────
def compute_suspicion(pc: list, ph: list, pe: list):
    """
    Returns (is_suspicious: bool, flags: list[str], entropy: float).
    Fires when the model is unusually certain AND both sub-models agree too closely.
    """
    flags    = []
    pred_idx = int(np.argmax(pe))

    entropy      = -sum(p * math.log(p + 1e-10) for p in pe)
    norm_entropy = entropy / math.log(len(pe))

    conf_ens    = pe[pred_idx]
    conf_cnn    = pc[pred_idx]
    conf_hybrid = ph[pred_idx]

    if conf_ens > 0.90 and norm_entropy < 0.12:
        flags.append('extremely_low_entropy')
    if abs(conf_cnn - conf_hybrid) < 0.02 and conf_ens > 0.88:
        flags.append('suspiciously_identical_outputs')
    if conf_ens > 0.95 and norm_entropy < 0.08:
        flags.append('near_zero_entropy')

    return len(flags) >= 2, flags, round(norm_entropy, 4)
