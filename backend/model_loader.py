"""
model_loader.py
───────────────
Downloads cnn_best.keras and hybrid_best.keras from Google Drive
on first boot, then caches them locally — subsequent restarts are instant.

Your actual model IDs are already embedded below.
Override with CNN_DRIVE_ID / HYBRID_DRIVE_ID env vars only if you
retrain and upload new versions.
"""

import os
import gdown

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
CNN_PATH    = os.path.join(MODELS_DIR, 'cnn_best.keras')
HYBRID_PATH = os.path.join(MODELS_DIR, 'hybrid_best.keras')

# ── Your actual trained model file IDs ───────────────────────────────
_DEFAULT_CNN_ID    = '1t2cJihSWhY0qnBXa_7xOBj1OvyMh_4Nw'
_DEFAULT_HYBRID_ID = '1gRD8hCYUYc7FylmlF-zGgYn7IND1lUkQ'


def _download(file_id: str, dest: str, name: str):
    """
    fuzzy=True handles Google Drive's large-file virus-scan
    confirmation page, which is the most common cause of failures.
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f'⬇  Downloading {name} from Google Drive …')
    gdown.download(url, dest, quiet=False, fuzzy=True)
    if not os.path.exists(dest) or os.path.getsize(dest) < 1024:
        raise RuntimeError(
            f'Download failed for {name}.\n'
            f'  Make sure the file is shared as "Anyone with the link".\n'
            f'  File ID used: {file_id}'
        )
    print(f'✅ {name} ready ({os.path.getsize(dest)/1_048_576:.1f} MB)')


def ensure_models():
    """
    Called once at startup.
    Returns (cnn_path, hybrid_path) — both guaranteed to exist on return.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    cnn_id    = os.environ.get('CNN_DRIVE_ID',    _DEFAULT_CNN_ID).strip()
    hybrid_id = os.environ.get('HYBRID_DRIVE_ID', _DEFAULT_HYBRID_ID).strip()

    if not os.path.exists(CNN_PATH):
        _download(cnn_id, CNN_PATH, 'cnn_best.keras')

    if not os.path.exists(HYBRID_PATH):
        _download(hybrid_id, HYBRID_PATH, 'hybrid_best.keras')

    return CNN_PATH, HYBRID_PATH
