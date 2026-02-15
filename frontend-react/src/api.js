const API = '/api';

export async function verify() {
  const res = await fetch(`${API}/verify`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function profile(opts = {}) {
  const res = await fetch(`${API}/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'resnet18', inferences: 20, batch_size: 8, ...opts }),
  });
  const data = await res.json();
  if (!data.success && data.error) throw new Error(data.error);
  return data;
}

export async function deduplicate(threshold = 0.8) {
  const res = await fetch(`${API}/deduplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold }),
  });
  const data = await res.json();
  if (!data.success && data.error) throw new Error(data.error);
  return data;
}

export async function evaluate() {
  const res = await fetch(`${API}/evaluate`, { method: 'POST' });
  const data = await res.json();
  if (!data.success && data.error) throw new Error(data.error);
  return data;
}

export async function impactSummary(profileData, dedupData, evalData) {
  const res = await fetch(`${API}/impact-summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      profile: profileData || null,
      dedup: dedupData || null,
      eval_data: evalData || null,
    }),
  });
  const data = await res.json();
  return data;
}
