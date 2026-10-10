const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(endpoint, options = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || `API request failed: ${response.status}`);
  }

  return data;
}

export const checkHealth = () => request("/health");

export const getModelInfo = () => request("/model-info");

export const predictChurn = (customerData) =>
  request("/predict", {
    method: "POST",
    body: JSON.stringify(customerData),
  });

export const predictBatch = (customers) =>
  request("/predict/batch", {
    method: "POST",
    body: JSON.stringify(customers),
  });