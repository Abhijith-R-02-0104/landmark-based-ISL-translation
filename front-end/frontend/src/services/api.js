const BASE_URL = "http://localhost:8000";

export const predictFrame = async (base64Image) => {
    try {
        const response = await fetch(`${BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64Image }),
        });
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Prediction error:", err);
        return null;
    }
};

export const checkBackendStatus = async () => {
    try {
        const response = await fetch(`${BASE_URL}/status`);
        return response.ok;
    } catch {
        return false;
    }
};