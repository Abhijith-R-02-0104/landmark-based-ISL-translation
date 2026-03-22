import React, {
  useRef,
  useState,
  forwardRef,
  useImperativeHandle
} from "react";
import Webcam from "react-webcam";

const WebcamView = forwardRef((props, ref) => {
  const webcamRef = useRef(null);
  const [prediction, setPrediction] = useState("");

  // expose capture to parent (App.jsx)
  useImperativeHandle(ref, () => ({
    capture() {
      if (!webcamRef.current) return null;

      const imageSrc = webcamRef.current.getScreenshot();
      return imageSrc;
    }
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        screenshotQuality={1}   // 🔥 HIGH QUALITY FRAME
        videoConstraints={{
          width: 640,          // 🔥 FIXED RESOLUTION
          height: 480,
          facingMode: "user"
        }}
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