const BASE_URL = import.meta.env.VITE_API_URL;

export const analyzeAlert = async (text) => {
  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  return res.json();
};