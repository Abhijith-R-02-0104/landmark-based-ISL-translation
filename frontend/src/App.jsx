import { useRef, useState, useEffect } from "react";
import WebcamView from "./components/WebcamView";
import { predictFrame } from "./services/api";

function App() {
  const webcamRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [predictedLetter, setPredictedLetter] = useState("-");
  const [currentWord, setCurrentWord] = useState("-");
  const [confidence, setConfidence] = useState(null);
    useEffect(() => {
    let interval;

    if (isDetecting) {
      interval = setInterval(async () => {
        const image = webcamRef.current.capture();

        if (!image) return;

        const result = await predictFrame(image);

        if (result) {
          setPredictedLetter(result.letter);
          setCurrentWord(result.current_word);
          setConfidence(result.confidence);
        } 
      }, 400);
    }

    return () => clearInterval(interval);
  }, [isDetecting]);
  return (
    <div className="min-h-screen bg-slate-900 text-white flex justify-center">
      <div className="w-full max-w-5xl px-6 py-10 space-y-10">

        {/* Header */}
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-wide">
            Dynamic Sign Language Recognition
          </h1>
          <p className="text-slate-400">
            Landmark-Based Temporal Hand Motion Analysis
          </p>
          <div className="border-t border-slate-700 pt-6" />
        </header>

        {/* Webcam */}
        <div className="flex justify-center">
          <div className="rounded-2xl overflow-hidden shadow-lg border border-slate-700">
            <WebcamView ref={webcamRef} />
          </div>
        </div>

        {/* Predictions */}
        <div className="flex justify-center gap-20 text-center">
          <div>
            <h2 className="text-lg text-slate-400">Predicted Letter</h2>

            <p className="text-5xl font-bold mt-2">{predictedLetter}</p>

            {confidence && (
            <p className="text-sm text-slate-400 mt-2">Confidence: {(confidence * 100).toFixed(1)}%</p>
            )}
          </div>

          <div>
            <h2 className="text-lg text-slate-400">Current Word</h2>
            <p className="text-3xl font-semibold mt-2">{currentWord}</p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex justify-center gap-6">
          <button onClick={() => setIsDetecting(true)} className="px-6 py-2 bg-green-600 rounded-lg hover:bg-green-700 transition">
            Start
          </button>

          <button onClick={() => setIsDetecting(false)} className="px-6 py-2 bg-red-600 rounded-lg hover:bg-red-700 transition">
            Stop
          </button>
          <button className="px-6 py-2 bg-slate-600 rounded-lg hover:bg-slate-700 transition">
            Clear
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;