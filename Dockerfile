# ── Stage 1: Build the React frontend ────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./frontend/
RUN npm install --prefix frontend --silent
COPY frontend/ ./frontend/
RUN npm run build --prefix frontend

# ── Stage 2: Python runtime ───────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# libgomp1     → required by tensorflow/numpy (OpenMP)
# libglib2.0-0 → required by some tensorflow internals
# curl         → useful for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built frontend + backend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
COPY backend/ ./backend/

ENV PORT=5000
EXPOSE 5000

CMD ["python3", "backend/app.py"]
