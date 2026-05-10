# LuffaGuard 🌿
**Luffa aegyptiaca Leaf Disease Detection** — CNN + CNN-ViT Hybrid Ensemble

Detects **Mosaic Disease** and **Insect Infestation** from leaf photos.
CNN 99.19% · Hybrid 99.51% accuracy.

---

## Deploy on Railway (3 steps)

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Add ONE environment variable:

| Variable | Value  |
|----------|--------|
| `PORT`   | `5000` |

That's it. On first boot the app auto-downloads your trained models
from Google Drive (~60–90 sec), then starts serving. All subsequent
restarts are instant (models cached in the container).

---

## Deploy on Render

1. Push to GitHub
2. [render.com](https://render.com) → New → Web Service → connect repo
3. Set:
   - **Build Command**: `npm run build`
   - **Start Command**: `npm start`
4. Add `PORT = 5000` in environment variables
5. Deploy

---

## Run locally

```bash
# Install Python deps
pip install -r requirements.txt

# Start (models download automatically on first run)
python backend/app.py
```

Open http://localhost:5000

If you already downloaded the .keras files from Drive:
```bash
mkdir models
cp /path/to/cnn_best.keras    models/
cp /path/to/hybrid_best.keras models/
python backend/app.py    # skips download, uses local files
```

---

## Project structure

```
luffaguard/
├── package.json          ← root build + start scripts
├── railway.toml          ← Railway config
├── render.yaml           ← Render config
├── requirements.txt      ← Python deps
├── backend/
│   ├── app.py            ← Flask (API + serves built frontend)
│   ├── model_loader.py   ← Auto-downloads models from Drive
│   └── disease_info.py   ← Disease knowledge base
└── frontend/             ← React + Vite
    └── src/
        ├── App.jsx        ← Full UI
        └── api.js
```

---

## Dataset
DOI: [10.17632/nym8bw5hr6.3](https://doi.org/10.17632/nym8bw5hr6.3)
