import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export async function predictFrame(imageBase64) {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      image: imageBase64,
    });

    return response.data;
  } catch (error) {
    console.error("Prediction error:", error);
    return null;
  }
}