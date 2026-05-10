const BASE = import.meta.env.DEV ? 'http://localhost:5000' : '';

export async function predictDisease(file) {
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(`${BASE}/api/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Server error' }));
    throw new Error(err.error || 'Prediction failed');
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/api/health`);
  return res.json();
}
