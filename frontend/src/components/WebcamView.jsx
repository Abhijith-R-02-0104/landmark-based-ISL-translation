import React, {
    useRef,
    useState,
    useEffect,
    forwardRef,
    useImperativeHandle
} from "react";
import Webcam from "react-webcam";

const WebcamView = forwardRef((props, ref) => {
    const webcamRef = useRef(null);
    const [prediction, setPrediction] = useState("");

  // Expose capture method to parent
    useImperativeHandle(ref, () => ({
    capture() {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
        sendFrameToBackend(imageSrc);
        }
        return imageSrc;
    }
    }));

    const sendFrameToBackend = async (base64Image) => {
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image: base64Image
        })
        });

        const data = await response.json();
        console.log("Prediction:", data);
        setPrediction(data.prediction || JSON.stringify(data));
    } catch (error) {
        console.error("Error sending frame:", error);
    }
    };

  // Automatically send frame every 1 second
    useEffect(() => {
    const interval = setInterval(() => {
        if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
            sendFrameToBackend(imageSrc);
        }
        }
    }, 100);

    return () => clearInterval(interval);
    }, []);

    return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }}
        style={{
            width: "480px",
            height: "360px",
            borderRadius: "16px",
            objectFit: "cover",
            transform: "scaleX(-1)"
        }}
        />

        <h2 style={{ marginTop: "16px" }}>
        Prediction: {prediction}
        </h2>
    </div>
    );
});

export default WebcamView;