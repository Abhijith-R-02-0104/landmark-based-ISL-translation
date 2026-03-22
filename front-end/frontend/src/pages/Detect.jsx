import { useRef, useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import ReactDOM from "react-dom";
import { predictFrame } from "../services/api";
import { useToast } from "../components/Toast";
import { usePageTitle } from "../App";

const CSS = `
  @keyframes camPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,232,184,0.5); }
    50%     { box-shadow: 0 0 0 8px rgba(16,232,184,0); }
  }
  @keyframes scanline {
    0%  { top: 0%;   opacity: 0.08; }
    50% {             opacity: 0.15; }
    100%{ top: 100%; opacity: 0.04; }
  }
  @keyframes letterPop {
    0%  { transform: scale(0.7); opacity: 0; }
    65% { transform: scale(1.08); }
    100%{ transform: scale(1);   opacity: 1; }
  }
  @keyframes wordFlash {
    0%  { box-shadow: none; border-color: var(--border); }
    35% { box-shadow: 0 0 28px 6px var(--accent-dim); border-color: var(--accent); background: var(--accent-dim); }
    100%{ box-shadow: none; border-color: var(--border); }
  }
  @keyframes wordScale { 0%{transform:scale(1)} 40%{transform:scale(1.06)} 100%{transform:scale(1)} }
  @keyframes spinDot   { to{transform:rotate(360deg)} }
  @keyframes cornerIn  { from{opacity:0;transform:scale(0.85)} to{opacity:1;transform:scale(1)} }
  @keyframes cursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes fsIn { from{opacity:0} to{opacity:1} }

  .letter-pop  { animation: letterPop  0.38s cubic-bezier(.22,.68,0,1.2) both; }
  .word-flash  { animation: wordFlash  0.75s ease both; }
  .word-scale  { animation: wordScale  0.4s ease both; }
  .corner-anim { animation: cornerIn   0.3s ease both; }

  .cam-btn { transition: all 0.22s ease; cursor: pointer; }
  .cam-btn:hover:not(:disabled) { transform: translateY(-2px); filter: brightness(1.1); }
  .cam-btn:active:not(:disabled){ transform: scale(0.96); }
  .cam-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .sentence-scroll {
    scrollbar-width: thin;
    scrollbar-color: var(--accent-border) var(--surface);
  }
  .sentence-scroll::-webkit-scrollbar { width: 5px; }
  .sentence-scroll::-webkit-scrollbar-track { background: var(--surface); border-radius: 999px; }
  .sentence-scroll::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 999px; }

  .typing-cursor::after { content: '|'; color: var(--accent); animation: cursorBlink 0.8s ease infinite; margin-left: 2px; }
  .block-card { transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease; }

  .fs-overlay {
    position: fixed; inset: 0; z-index: 99999;
    background: #000; display: flex; flex-direction: column;
    animation: fsIn 0.3s ease both;
  }
  .fs-exit-btn {
    position: absolute; top: 20px; right: 20px;
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.15); color: #e8f4ff;
    cursor: pointer; transition: all 0.22s ease;
  }
  .fs-exit-btn:hover { border-color: rgba(239,68,68,0.5); color: #ef4444; transform: scale(1.05); }
  .fs-results {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(0deg, rgba(5,8,15,0.97) 0%, rgba(5,8,15,0.7) 100%);
    backdrop-filter: blur(12px); padding: 20px 28px;
    display: grid; grid-template-columns: auto 1fr auto auto;
    align-items: center; gap: 20px;
    border-top: 1px solid rgba(16,232,184,0.15);
  }
  .copy-btn { transition: all 0.22s ease; }
  .copy-btn:hover { border-color: var(--accent-border) !important; background: var(--accent-dim) !important; color: var(--accent) !important; }
`;

export default function Detect() {
    usePageTitle("Detection");
    const navigate = useNavigate();
    const toast = useToast();
    const videoRef = useRef(null);
    const fsVideoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const typingRef = useRef(null);
    const confAnimRef = useRef(null);
    const confCurRef = useRef(0);
    const prevSentRef = useRef("");
    const prevWordRef = useRef("");

    const [cameraActive, setCameraActive] = useState(false);
    const [isDetecting, setIsDetecting] = useState(false);
    const [isPredicting, setIsPredicting] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [predictedLetter, setPredictedLetter] = useState("—");
    const [letterKey, setLetterKey] = useState(0);
    const [confidence, setConfidence] = useState(0);
    const [displayConf, setDisplayConf] = useState(0);
    const [currentWord, setCurrentWord] = useState("");
    const [wordFlash, setWordFlash] = useState(false);
    const [wordKey, setWordKey] = useState(0);
    const [typedSentence, setTypedSentence] = useState("");
    const [sentence, setSentence] = useState([]);
    const [error, setError] = useState(null);

    // ── Camera ────────────────────────────────────────────────
    const startCamera = async () => {
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
            streamRef.current = stream;
            if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
            setCameraActive(true);
            toast.success("Camera started", { title: "Camera Active" });
        } catch {
            setError("Camera access denied — please allow camera permissions.");
            toast.error("Could not access camera", { title: "Camera Error" });
        }
    };

    const stopCamera = () => {
        setIsDetecting(false); setIsPredicting(false); setIsFullscreen(false);
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;
        setCameraActive(false);
        toast.info("Camera stopped", { title: "Camera Off" });
    };

    const captureFrame = useCallback(() => {
        const v = videoRef.current, c = canvasRef.current;
        if (!v || !c || v.readyState < 2) return null;
        c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
        c.getContext("2d").drawImage(v, 0, 0);
        return c.toDataURL("image/jpeg", 0.8).split(",")[1];
    }, []);

    // ── Detection Loop ─────────────────────────────────────────
    useEffect(() => {
        let interval = null;
        if (isDetecting) {
            interval = setInterval(async () => {
                const frame = captureFrame();
                if (!frame) return;
                setIsPredicting(true);
                const result = await predictFrame(frame);
                setIsPredicting(false);
                if (result) {
                    const letter = result.letter || "—";
                    setPredictedLetter(prev => { if (prev !== letter) setLetterKey(k => k + 1); return letter; });
                    setConfidence(result.confidence || 0);
                    setCurrentWord(result.current_word || "");
                    if (result.current_word) {
                        setSentence(prev => {
                            const last = prev[prev.length - 1];
                            return last !== result.current_word ? [...prev, result.current_word] : prev;
                        });
                    }
                }
            }, 600);
        }
        return () => { if (interval) clearInterval(interval); };
    }, [isDetecting, captureFrame]);

    // ── Smooth Confidence ─────────────────────────────────────
    const confPct = Math.round(confidence > 1 ? confidence : confidence * 100);
    useEffect(() => {
        if (confAnimRef.current) cancelAnimationFrame(confAnimRef.current);
        const start = confCurRef.current, target = confPct;
        const startTime = performance.now(), duration = 450;
        const animate = (now) => {
            const p = Math.min((now - startTime) / duration, 1);
            const val = Math.round(start + (target - start) * (1 - Math.pow(1 - p, 3)));
            confCurRef.current = val; setDisplayConf(val);
            if (p < 1) confAnimRef.current = requestAnimationFrame(animate);
        };
        confAnimRef.current = requestAnimationFrame(animate);
        return () => { if (confAnimRef.current) cancelAnimationFrame(confAnimRef.current); };
    }, [confPct]);

    // ── Word Flash ────────────────────────────────────────────
    useEffect(() => {
        if (currentWord && currentWord !== prevWordRef.current) {
            prevWordRef.current = currentWord;
            setWordFlash(true); setWordKey(k => k + 1);
            const t = setTimeout(() => setWordFlash(false), 750);
            return () => clearTimeout(t);
        }
    }, [currentWord]);

    // ── Typewriter ────────────────────────────────────────────
    useEffect(() => {
        const full = sentence.join(" ");
        if (full === prevSentRef.current) return;
        const newPart = full.slice(prevSentRef.current.length);
        prevSentRef.current = full;
        if (typingRef.current) clearInterval(typingRef.current);
        let i = 0;
        typingRef.current = setInterval(() => {
            if (i < newPart.length) { setTypedSentence(p => p + newPart[i]); i++; }
            else clearInterval(typingRef.current);
        }, 42);
        return () => clearInterval(typingRef.current);
    }, [sentence]);

    // ── Re-attach stream when toggling fullscreen ─────────────
    useEffect(() => {
        if (streamRef.current) {
            if (videoRef.current) {
                videoRef.current.srcObject = streamRef.current;
                videoRef.current.play().catch(() => { });
            }
            if (fsVideoRef.current) {
                fsVideoRef.current.srcObject = streamRef.current;
                fsVideoRef.current.play().catch(() => { });
            }
        }
    }, [isFullscreen]);

    // ── Actions ───────────────────────────────────────────────
    const handleStartDetection = () => {
        setIsDetecting(true);
        toast.success("Detection is running", { title: "Detection Started" });
    };
    const handleStopDetection = () => {
        setIsDetecting(false); setIsPredicting(false);
        toast.warning("Detection has stopped", { title: "Detection Stopped" });
    };
    const handleClear = () => {
        setPredictedLetter("—"); setConfidence(0); setDisplayConf(0);
        confCurRef.current = 0;
        setCurrentWord(""); setSentence([]); setTypedSentence("");
        prevSentRef.current = ""; prevWordRef.current = "";
        toast.info("All results cleared");
    };
    const handleCopySentence = () => {
        if (!typedSentence) { toast.warning("No sentence to copy yet"); return; }
        navigator.clipboard.writeText(typedSentence)
            .then(() => toast.success("Sentence copied to clipboard!", { title: "Copied!" }))
            .catch(() => toast.error("Could not copy — try manually"));
    };

    // ── Text-to-Speech ────────────────────────────────────────
    const handleSpeak = () => {
        if (!typedSentence) { toast.warning("No sentence to read yet"); return; }
        if (!window.speechSynthesis) { toast.error("Speech not supported in this browser"); return; }
        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(typedSentence);
        utt.rate = 0.95; utt.pitch = 1; utt.volume = 1;
        utt.onstart = () => toast.info("Reading sentence aloud...", { title: "Speaking" });
        utt.onerror = () => toast.error("Speech failed — try again");
        window.speechSynthesis.speak(utt);
    };

    // ── Network status toast when detecting ──────────────────
    const wasOnlineRef = useRef(true);
    useEffect(() => {
        if (!isDetecting) return;
        const check = setInterval(async () => {
            try {
                const res = await fetch("http://localhost:8000/status");
                const online = res.ok;
                if (!online && wasOnlineRef.current) {
                    toast.error("Backend went offline during detection!", { title: "Connection Lost" });
                    wasOnlineRef.current = false;
                }
                if (online && !wasOnlineRef.current) {
                    toast.success("Backend reconnected!", { title: "Reconnected" });
                    wasOnlineRef.current = true;
                }
            } catch {
                if (wasOnlineRef.current) {
                    toast.error("Backend went offline during detection!", { title: "Connection Lost" });
                    wasOnlineRef.current = false;
                }
            }
        }, 4000);
        return () => clearInterval(check);
    }, [isDetecting]);

    // ── ESC to exit fullscreen ────────────────────────────────
    useEffect(() => {
        const onKey = (e) => { if (e.key === "Escape") setIsFullscreen(false); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, []);

    const confZone = displayConf >= 80
        ? { label: "High", color: "var(--accent)", bar: "linear-gradient(90deg,var(--accent),#0abf97)" }
        : displayConf >= 50
            ? { label: "Medium", color: "#f59e0b", bar: "linear-gradient(90deg,#f59e0b,#d97706)" }
            : { label: "Low", color: "#ef4444", bar: "linear-gradient(90deg,#ef4444,#dc2626)" };

    const statusLabel = !cameraActive ? "Camera Off" : isDetecting ? (isPredicting ? "Analyzing..." : "Detecting...") : "Camera Ready";
    const statusColor = !cameraActive ? "var(--text-faint)" : isDetecting ? "var(--accent)" : "var(--text-muted)";

    // ── FULLSCREEN — rendered via Portal directly to body ─────
    const fullscreenEl = isFullscreen && cameraActive ? (
        <div className="fs-overlay">
            <style>{CSS}</style>
            <canvas ref={canvasRef} style={{ display: "none" }} />
            <video ref={fsVideoRef} style={{ width: "100%", height: "100%", objectFit: "contain", transform: "scaleX(-1)" }} muted playsInline autoPlay />

            {isDetecting && <div style={{ position: "absolute", left: 0, right: 0, height: "2px", background: "linear-gradient(90deg,transparent,rgba(16,232,184,0.5),transparent)", animation: "scanline 2.8s ease-in-out infinite", pointerEvents: "none" }} />}
            {isDetecting && (
                <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                    {[{ top: "20px", left: "20px", borderRight: "none", borderBottom: "none" }, { top: "20px", right: "20px", borderLeft: "none", borderBottom: "none" }, { bottom: "120px", left: "20px", borderRight: "none", borderTop: "none" }, { bottom: "120px", right: "20px", borderLeft: "none", borderTop: "none" }].map((st, i) => (
                        <div key={i} style={{ position: "absolute", width: "32px", height: "32px", border: "2px solid var(--accent)", opacity: 0.85, ...st }} />
                    ))}
                </div>
            )}
            {isDetecting && (
                <div style={{ position: "absolute", top: "20px", left: "20px", display: "inline-flex", alignItems: "center", gap: "6px", padding: "6px 14px", borderRadius: "999px", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)", border: "1px solid rgba(16,232,184,0.35)", fontSize: "11px", fontWeight: 700, letterSpacing: "0.1em", color: "var(--accent)", fontFamily: "'Space Mono',monospace" }}>
                    <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "camPulse 1.4s ease infinite", display: "inline-block" }} /> LIVE
                </div>
            )}
            <button className="fs-exit-btn" onClick={() => setIsFullscreen(false)} title="Exit (ESC)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M8 3v3a2 2 0 01-2 2H3m18 0h-3a2 2 0 01-2-2V3m0 18v-3a2 2 0 012-2h3M3 16h3a2 2 0 012 2v3" />
                </svg>
            </button>
            <div className="fs-results">
                <div>
                    <div style={{ fontSize: "0.65rem", color: "rgba(232,244,255,0.4)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "4px", fontFamily: "'Space Mono',monospace" }}>Sign</div>
                    <div key={letterKey} className="letter-pop" style={{ fontFamily: "'Space Mono',monospace", fontSize: "3rem", fontWeight: 800, color: "var(--accent)", textShadow: "0 0 24px rgba(16,232,184,0.5)", lineHeight: 1 }}>{predictedLetter}</div>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
                        <span style={{ fontSize: "0.72rem", color: "rgba(232,244,255,0.4)", textTransform: "uppercase", letterSpacing: "0.07em", fontFamily: "'Space Mono',monospace" }}>Confidence</span>
                        <span style={{ color: confZone.color, fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.85rem" }}>{displayConf}%</span>
                        <span style={{ padding: "2px 8px", borderRadius: "999px", background: `${confZone.color}18`, border: `1px solid ${confZone.color}40`, fontSize: "0.68rem", fontWeight: 700, color: confZone.color }}>{confZone.label}</span>
                    </div>
                    <div style={{ height: "6px", borderRadius: "999px", background: "rgba(255,255,255,0.1)", overflow: "hidden", marginBottom: "10px" }}>
                        <div style={{ height: "100%", borderRadius: "999px", width: `${displayConf}%`, background: confZone.bar, transition: "width 0.08s linear" }} />
                    </div>
                    <div style={{ fontSize: "0.88rem", color: "rgba(232,244,255,0.75)", fontFamily: "'Space Mono',monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {typedSentence || <span style={{ opacity: 0.35 }}>Sentence will appear here...</span>}
                    </div>
                </div>
                <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "0.65rem", color: "rgba(232,244,255,0.4)", textTransform: "uppercase", letterSpacing: "0.07em", fontFamily: "'Space Mono',monospace", marginBottom: "4px" }}>Word</div>
                    <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "1.3rem", fontWeight: 700, color: "#e8f4ff" }}>{currentWord || "—"}</div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    <button className="cam-btn micro-btn" style={{ ...d.btnTeal, flex: "none", padding: "10px 18px", fontSize: "0.82rem" }} disabled={!cameraActive}
                        onClick={isDetecting ? handleStopDetection : handleStartDetection}>
                        {isDetecting ? "Stop" : "Start"}
                    </button>
                    <button className="cam-btn micro-btn" style={{ ...d.btnGhost, flex: "none", padding: "10px 18px", fontSize: "0.82rem" }} onClick={handleClear}>Clear</button>
                </div>
            </div>
        </div>
    ) : null;

    // ── NORMAL VIEW ────────────────────────────────────────────
    return (
        <>
            {/* Portal renders fullscreen DIRECTLY to body — above everything */}
            {fullscreenEl && ReactDOM.createPortal(fullscreenEl, document.body)}

            <main style={d.page}>
                <style>{CSS}</style>
                <canvas ref={canvasRef} style={{ display: "none" }} />

                <div style={d.topBar}>
                    <div>
                        <button style={d.backBtn} onClick={() => navigate("/")}>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
                            Back to Home
                        </button>
                        <h1 style={d.pageTitle}>Detection Console</h1>
                    </div>
                    <div style={{ ...d.statusPill, borderColor: isDetecting ? "var(--accent-border)" : "var(--border)", color: statusColor }}>
                        <span style={{ ...d.dot, background: statusColor, boxShadow: isDetecting ? "0 0 8px var(--accent)" : "none", animation: isDetecting ? "camPulse 1.5s ease infinite" : "none" }} />
                        {statusLabel}
                    </div>
                </div>

                {error && <div style={d.errorBar}>⚠ {error}</div>}

                <div style={d.grid}>
                    {/* Camera */}
                    <div style={d.camPanel}>
                        <div style={d.panelHead}>
                            <span style={d.panelLabel}>Webcam Feed</span>
                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                                {cameraActive && (
                                    <button className="cam-btn micro-btn" style={d.fsBtn} onClick={() => setIsFullscreen(true)} title="Fullscreen">
                                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3" />
                                        </svg>
                                        Fullscreen
                                    </button>
                                )}
                                <span style={{ ...d.camLed, background: cameraActive ? "var(--accent)" : "var(--text-faint)", boxShadow: cameraActive ? "0 0 10px var(--accent)" : "none" }} />
                            </div>
                        </div>
                        <div style={d.camBox}>
                            <video ref={videoRef} style={{ ...d.video, opacity: cameraActive ? 1 : 0 }} muted playsInline />
                            {!cameraActive && (
                                <div style={d.camIdle}>
                                    <div style={d.camIconWrap}>
                                        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.4" opacity="0.5">
                                            <path d="M23 7l-7 5 7 5V7z" /><rect x="1" y="5" width="15" height="14" rx="2" />
                                        </svg>
                                    </div>
                                    <p style={d.idleText}>Camera not active</p>
                                    <p style={d.idleSub}>Click Start Camera below</p>
                                </div>
                            )}
                            {isDetecting && cameraActive && (
                                <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                                    {[{ top: "14px", left: "14px", borderRight: "none", borderBottom: "none" }, { top: "14px", right: "14px", borderLeft: "none", borderBottom: "none" }, { bottom: "14px", left: "14px", borderRight: "none", borderTop: "none" }, { bottom: "14px", right: "14px", borderLeft: "none", borderTop: "none" }].map((st, i) => (
                                        <div key={i} className="corner-anim" style={{ position: "absolute", width: "26px", height: "26px", border: "2px solid var(--accent)", opacity: 0.85, animationDelay: `${i * 0.06}s`, ...st }} />
                                    ))}
                                </div>
                            )}
                            {isDetecting && <div style={d.scanline} />}
                            {isDetecting && (
                                <div style={d.liveBadge}>
                                    <span style={{ ...d.dot, background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "camPulse 1.4s ease infinite" }} /> LIVE
                                </div>
                            )}
                            {isDetecting && predictedLetter !== "—" && (
                                <div style={d.letterOverlay}>
                                    <span key={letterKey} className="letter-pop" style={d.letterOverlayText}>{predictedLetter}</span>
                                </div>
                            )}
                        </div>
                        <div style={d.camFooter}>
                            {!cameraActive
                                ? <button className="cam-btn micro-btn" style={d.btnTeal} onClick={startCamera}>
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M23 7l-7 5 7 5V7zM1 5h15a2 2 0 012 2v10a2 2 0 01-2 2H1a2 2 0 01-2-2V7a2 2 0 012-2z" /></svg>
                                    Start Camera
                                </button>
                                : <button className="cam-btn micro-btn" style={d.btnGhost} onClick={stopCamera}>
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" /></svg>
                                    Stop Camera
                                </button>
                            }
                        </div>
                    </div>

                    {/* Right column */}
                    <div style={d.rightCol}>
                        <div className="block-card glass-card" style={{ ...d.block, ...(isDetecting ? d.blockActive : {}) }}>
                            <div style={d.blockHead}>
                                <span style={d.blockLabel}>Detected Sign</span>
                                {isPredicting && <span style={d.spinnerWrap}><span style={d.spinnerDot} /> analyzing</span>}
                            </div>
                            <div key={letterKey} className="letter-pop" style={d.bigLetter}>{predictedLetter}</div>
                        </div>

                        <div className="block-card glass-card" style={{ ...d.block, ...(isDetecting ? d.blockActive : {}) }}>
                            <div style={d.blockHead}>
                                <span style={d.blockLabel}>Confidence</span>
                                <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                    <span style={{ ...d.zoneTag, background: `${confZone.color}18`, color: confZone.color, border: `1px solid ${confZone.color}40` }}>{confZone.label}</span>
                                    <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.95rem", color: confZone.color, fontWeight: 700 }}>{displayConf}%</span>
                                </span>
                            </div>
                            <div style={d.barTrack}>
                                <div style={{ ...d.barFill, width: `${displayConf}%`, background: confZone.bar }} />
                            </div>
                            <div style={d.barZones}>
                                <span style={{ color: "#ef4444" }}>Low</span>
                                <span style={{ color: "#f59e0b" }}>Medium</span>
                                <span style={{ color: "var(--accent)" }}>High</span>
                            </div>
                        </div>

                        <div key={wordKey} className={`block-card glass-card${wordFlash ? " word-flash" : ""}`} style={{ ...d.block, ...(!wordFlash && isDetecting ? d.blockActive : {}) }}>
                            <span style={d.blockLabel}>Current Word</span>
                            <div key={wordKey} className={wordFlash ? "word-scale" : ""} style={d.wordDisplay}>
                                {currentWord || <span style={{ color: "var(--text-faint)" }}>waiting...</span>}
                            </div>
                        </div>

                        <div className="block-card glass-card" style={d.block}>
                            <div style={d.blockHead}>
                                <span style={d.blockLabel}>Sentence</span>
                                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                    <span style={{ fontSize: "0.72rem", color: "var(--text-faint)", fontFamily: "'Space Mono',monospace" }}>{sentence.length} words</span>
                                    <button className="copy-btn cam-btn micro-btn" style={{ ...d.copyBtn }} onClick={handleCopySentence} title="Copy sentence">
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                                        </svg>
                                        Copy
                                    </button>
                                    <button className="copy-btn cam-btn micro-btn" style={{ ...d.copyBtn }} onClick={handleSpeak} title="Read sentence aloud">
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M15.54 8.46a5 5 0 010 7.07" /><path d="M19.07 4.93a10 10 0 010 14.14" />
                                        </svg>
                                        Speak
                                    </button>
                                </div>
                            </div>
                            <div className={`sentence-scroll${typedSentence ? " typing-cursor" : ""}`} style={d.sentenceBox}>
                                {typedSentence || <span style={{ color: "var(--text-faint)" }}>Sentence will appear here...</span>}
                            </div>
                        </div>

                        <div style={d.controlRow}>
                            <button className="cam-btn micro-btn" style={isDetecting ? d.btnRed : d.btnTeal} disabled={!cameraActive}
                                onClick={isDetecting ? handleStopDetection : handleStartDetection}>
                                {isDetecting
                                    ? <><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" /></svg> Stop Detection</>
                                    : <><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21" /></svg> Start Detection</>
                                }
                            </button>
                            <button className="cam-btn micro-btn" style={d.btnGhost} onClick={handleClear}>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.51" /></svg>
                                Clear
                            </button>
                        </div>

                        <div style={d.modelCard}>
                            <div style={d.modelTitle}>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" /></svg>
                                Model Info
                            </div>
                            <div style={d.modelGrid}>
                                {[["Architecture", "GRU (Keras)"], ["Input Shape", "30 × 63"], ["Landmarks", "21 × 3D (x,y,z)"], ["Accuracy", "85 – 95%"], ["Framework", "TensorFlow"], ["Backbone", "MediaPipe"]].map(([k, v]) => (
                                    <div key={k} style={d.modelRow}>
                                        <span style={d.modelKey}>{k}</span>
                                        <span style={d.modelVal}>{v}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}

const d = {
    page: { position: "relative", zIndex: 1, maxWidth: "1300px", margin: "0 auto", padding: "0 24px 60px", fontFamily: "'Plus Jakarta Sans',system-ui,sans-serif", color: "var(--text)" },
    topBar: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px", paddingBottom: "28px" },
    backBtn: { display: "inline-flex", alignItems: "center", gap: "7px", background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: "0.85rem", padding: 0, fontFamily: "inherit", marginBottom: "8px" },
    pageTitle: { margin: 0, fontSize: "clamp(1.5rem,3vw,2.3rem)", fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text)" },
    statusPill: { display: "inline-flex", alignItems: "center", gap: "7px", padding: "9px 16px", borderRadius: "999px", background: "var(--surface-2)", border: "1px solid var(--border)", fontSize: "0.82rem", fontWeight: 600, fontFamily: "'Space Mono',monospace", letterSpacing: "0.04em" },
    dot: { width: "7px", height: "7px", borderRadius: "50%", flexShrink: 0 },
    errorBar: { padding: "12px 18px", borderRadius: "12px", marginBottom: "18px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)", color: "#fca5a5", fontSize: "0.88rem" },
    grid: { display: "grid", gridTemplateColumns: "1.45fr 1fr", gap: "20px", alignItems: "start" },
    camPanel: { borderRadius: "22px", overflow: "hidden", background: "var(--surface)", border: "1px solid var(--border)" },
    panelHead: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 18px", borderBottom: "1px solid var(--border-sub)" },
    panelLabel: { fontSize: "0.78rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" },
    camLed: { width: "9px", height: "9px", borderRadius: "50%", transition: "all 0.4s ease" },
    fsBtn: { display: "inline-flex", alignItems: "center", gap: "6px", padding: "6px 12px", borderRadius: "9px", border: "1px solid var(--accent-border)", background: "var(--accent-dim)", color: "var(--accent)", fontWeight: 600, fontSize: "0.75rem", fontFamily: "inherit" },
    camBox: { position: "relative", background: "#020508", minHeight: "460px", overflow: "hidden" },
    video: { width: "100%", height: "460px", objectFit: "cover", display: "block", transition: "opacity 0.4s ease", transform: "scaleX(-1)" },
    camIdle: { position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "8px" },
    camIconWrap: { width: "88px", height: "88px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", marginBottom: "8px" },
    idleText: { margin: 0, fontSize: "0.95rem", fontWeight: 600, color: "var(--text-muted)" },
    idleSub: { margin: 0, fontSize: "0.82rem", color: "var(--text-faint)" },
    scanline: { position: "absolute", left: 0, right: 0, height: "2px", background: "linear-gradient(90deg,transparent,rgba(16,232,184,0.5),transparent)", animation: "scanline 2.8s ease-in-out infinite", pointerEvents: "none" },
    liveBadge: { position: "absolute", top: "13px", left: "13px", display: "inline-flex", alignItems: "center", gap: "6px", padding: "5px 12px", borderRadius: "999px", background: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)", border: "1px solid var(--accent-border)", fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", color: "var(--accent)", fontFamily: "'Space Mono',monospace" },
    letterOverlay: { position: "absolute", bottom: "18px", right: "18px", padding: "10px 16px", borderRadius: "12px", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)", border: "1px solid var(--accent-border)" },
    letterOverlayText: { fontFamily: "'Space Mono',monospace", fontSize: "2rem", fontWeight: 700, color: "var(--accent)", textShadow: "0 0 20px var(--accent-dim)" },
    camFooter: { display: "flex", gap: "12px", padding: "14px 18px", borderTop: "1px solid var(--border-sub)" },
    rightCol: { display: "flex", flexDirection: "column", gap: "12px" },
    block: { padding: "16px 18px", borderRadius: "16px", background: "var(--surface)", border: "1px solid var(--border)" },
    blockActive: { borderColor: "var(--accent-border)", boxShadow: "0 0 24px var(--accent-dim)" },
    blockHead: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" },
    blockLabel: { fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" },
    bigLetter: { fontSize: "4rem", fontWeight: 800, lineHeight: 1, fontFamily: "'Space Mono',monospace", color: "var(--accent)", textShadow: "0 0 28px var(--accent-dim)" },
    wordDisplay: { fontSize: "1.55rem", fontWeight: 700, fontFamily: "'Space Mono',monospace", color: "var(--text)", marginTop: "4px", minHeight: "2rem" },
    barTrack: { height: "8px", borderRadius: "999px", background: "var(--surface-2)", overflow: "hidden", marginBottom: "8px" },
    barFill: { height: "100%", borderRadius: "999px", transition: "width 0.08s linear, background 0.45s ease", minWidth: "4px" },
    barZones: { display: "flex", justifyContent: "space-between", fontSize: "0.7rem", opacity: 0.55 },
    zoneTag: { padding: "3px 9px", borderRadius: "999px", fontSize: "0.72rem", fontWeight: 700, fontFamily: "'Space Mono',monospace", letterSpacing: "0.04em" },
    sentenceBox: { minHeight: "64px", maxHeight: "110px", overflowY: "auto", marginTop: "6px", fontSize: "0.95rem", lineHeight: 1.75, color: "var(--text)", fontFamily: "'Space Mono',monospace", wordBreak: "break-word" },
    copyBtn: { display: "inline-flex", alignItems: "center", gap: "5px", padding: "5px 10px", borderRadius: "8px", border: "1px solid var(--border)", background: "var(--surface-2)", color: "var(--text-muted)", fontSize: "0.72rem", fontWeight: 600, fontFamily: "inherit" },
    spinnerWrap: { display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "0.75rem", color: "var(--accent)", fontFamily: "'Space Mono',monospace" },
    spinnerDot: { display: "inline-block", width: "11px", height: "11px", borderRadius: "50%", border: "2px solid var(--accent-dim)", borderTopColor: "var(--accent)", animation: "spinDot 0.7s linear infinite" },
    btnTeal: { flex: 1, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "11px 18px", borderRadius: "12px", border: "none", background: "linear-gradient(135deg,var(--accent),#0abf97)", color: "var(--accent-btn-text)", fontWeight: 700, fontSize: "0.88rem", fontFamily: "inherit", boxShadow: "0 6px 20px var(--accent-dim)" },
    btnRed: { flex: 1, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "11px 18px", borderRadius: "12px", border: "none", background: "linear-gradient(135deg,#ef4444,#dc2626)", color: "#fff", fontWeight: 700, fontSize: "0.88rem", fontFamily: "inherit", boxShadow: "0 6px 20px rgba(239,68,68,0.2)" },
    btnGhost: { flex: 1, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "11px 18px", borderRadius: "12px", border: "1px solid var(--border)", background: "var(--surface-2)", color: "var(--text-muted)", fontWeight: 600, fontSize: "0.88rem", fontFamily: "inherit" },
    controlRow: { display: "flex", gap: "10px" },
    modelCard: { padding: "16px 18px", borderRadius: "16px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)" },
    modelTitle: { display: "flex", alignItems: "center", gap: "7px", fontSize: "0.78rem", fontWeight: 700, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "12px" },
    modelGrid: { display: "flex", flexDirection: "column", gap: "6px" },
    modelRow: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "5px 0", borderBottom: "1px solid var(--border-sub)" },
    modelKey: { fontSize: "0.78rem", color: "var(--text-muted)" },
    modelVal: { fontSize: "0.78rem", fontWeight: 600, color: "var(--text)", fontFamily: "'Space Mono',monospace" },
};