import axios from 'axios';

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeAlert = (text) => {
  return API.post('/analyze', { text }).then((r) => r.data);
};
