const BASE = import.meta.env.DEV ? 'http://localhost:5000' : '';

export async function predictDisease(file) {
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(`${BASE}/api/predict`, { method:'POST', body:form });
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.error||'Server error'); }
  return res.json();
}

export async function predictBatch(files) {
  const form = new FormData();
  Array.from(files).forEach(f => form.append('images', f));
  const res = await fetch(`${BASE}/api/predict-batch`, { method:'POST', body:form });
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.error||'Server error'); }
  return res.json();
}

export async function checkHealth() {
  return fetch(`${BASE}/api/health`).then(r=>r.json());
}
