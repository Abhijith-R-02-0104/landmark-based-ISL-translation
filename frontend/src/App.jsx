import { useRef, useState, useEffect } from "react";
import WebcamView from "./components/WebcamView";
import { predictFrame } from "./services/api";

function App() {

  const webcamRef = useRef(null);

  const [isDetecting, setIsDetecting] = useState(false);
  const [predictedLetter, setPredictedLetter] = useState("-");
  const [currentWord, setCurrentWord] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [lastSpoken, setLastSpoken] = useState("");

  // ================= TEXT TO SPEECH =================
  const speakSentence = (text) => {

    const speech = new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 1;

    window.speechSynthesis.speak(speech);
  };

  // ================= DETECTION LOOP =================
  useEffect(() => {

    let interval;

    if (isDetecting) {

      interval = setInterval(async () => {

        if (!webcamRef.current) return;

        const image = webcamRef.current.capture();
        if (!image) return;

        const result = await predictFrame(image);

        if (result) {

          setPredictedLetter(result.letter || "-");
          setConfidence(result.confidence || 0);

          // sentence update
          if (result.current_word !== lastSpoken) {

            setCurrentWord(result.current_word);

            if (result.current_word !== "") {

              speakSentence(result.current_word);

              setLastSpoken(result.current_word);
            }
          }
        }

      }, 600); // capture every 600ms

    }

    return () => clearInterval(interval);

  }, [isDetecting, lastSpoken]);

  // ================= CLEAR SENTENCE =================
  const clearSentence = () => {

    setCurrentWord("");
    setLastSpoken("");
    setPredictedLetter("-");
    setConfidence(0);

  };

  return (

    <div className="min-h-screen bg-slate-900 text-white flex justify-center">

      <div className="w-full max-w-5xl px-6 py-10 space-y-10">

        {/* HEADER */}
        <header className="text-center space-y-2">

          <h1 className="text-4xl font-bold tracking-wide">
            Dynamic Sign Language Recognition
          </h1>

          <p className="text-slate-400">
            Landmark-Based Temporal Hand Motion Analysis
          </p>

          <div className="border-t border-slate-700 pt-6" />

        </header>

        {/* WEBCAM */}
        <div className="flex justify-center">

          <div className="rounded-2xl overflow-hidden shadow-lg border border-slate-700">
            <WebcamView ref={webcamRef} />
          </div>

        </div>

        {/* STATUS + CONFIDENCE */}
        <div className="text-center space-y-6">

          {/* STATUS */}
          <div>

            <h2 className="text-lg text-slate-400">
              Gesture Status
            </h2>

            {predictedLetter === "-" ? (

              <p className="text-yellow-400 text-xl mt-2">
                Detecting...
              </p>

            ) : (

              <p className="text-green-400 text-2xl font-bold mt-2">
                Confirmed: {predictedLetter}
              </p>

            )}

          </div>

          {/* CONFIDENCE BAR */}
          <div className="w-full max-w-md mx-auto">

            <h2 className="text-lg text-slate-400 mb-2">
              Confidence
            </h2>

            <div className="w-full bg-slate-700 rounded-full h-4">

              <div
                className="bg-green-500 h-4 rounded-full transition-all"
                style={{ width: `${confidence * 100}%` }}
              />

            </div>

            <p className="text-sm text-slate-400 mt-1">
              {(confidence * 100).toFixed(1)}%
            </p>

          </div>

        </div>

        {/* SENTENCE DISPLAY */}
        <div className="flex justify-center">

          <div className="w-full max-w-xl">

            <h2 className="text-lg text-slate-400 mb-2 text-center">
              Detected Sentence
            </h2>

            <div className="bg-slate-800 border border-slate-600 rounded-xl p-4 text-center text-2xl font-semibold h-20 overflow-x-auto whitespace-nowrap">

              {currentWord || "..."}

            </div>

          </div>

        </div>

        {/* CONTROLS */}
        <div className="flex justify-center gap-6">

          <button
            onClick={() => setIsDetecting(true)}
            className="px-6 py-2 bg-green-600 rounded-lg hover:bg-green-700 transition"
          >
            Start
          </button>

          <button
            onClick={() => setIsDetecting(false)}
            className="px-6 py-2 bg-red-600 rounded-lg hover:bg-red-700 transition"
          >
            Stop
          </button>

          <button
            onClick={clearSentence}
            className="px-6 py-2 bg-slate-600 rounded-lg hover:bg-slate-700 transition"
          >
            Clear Sentence
          </button>

        </div>

      </div>

    </div>
  );
}

export default App;