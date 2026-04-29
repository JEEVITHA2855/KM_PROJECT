const BASE_URL = "https://alert-classification-system-kmrl.onrender.com";

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