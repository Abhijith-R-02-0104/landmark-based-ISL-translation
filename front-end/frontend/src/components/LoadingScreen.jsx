import { useState, useEffect } from "react";
import LogoIcon from "./LogoIcon";

const CSS = `
  @keyframes cursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes subtitleFade { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes screenFade   { from{opacity:1} to{opacity:0} }
  @keyframes dotPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,232,184,0.6); }
    50%     { box-shadow: 0 0 0 10px rgba(16,232,184,0); }
  }
  @keyframes logoEntrance {
    0%   { opacity: 0; transform: scale(0.82) translateY(12px); }
    60%  { opacity: 1; transform: scale(1.04) translateY(-2px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes glowRing {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,232,184,0.25); }
    50%     { box-shadow: 0 0 0 14px rgba(16,232,184,0); }
  }

  .ls-logo-wrap {
    animation: logoEntrance 0.7s cubic-bezier(.22,.68,0,1.2) 0.1s both;
  }
  .ls-glow {
    border-radius: 24px;
    animation: glowRing 2.5s ease-in-out infinite;
  }
  .ls-cursor {
    display: inline-block;
    width: 3px; height: 0.85em;
    background: #10e8b8;
    margin-left: 4px;
    vertical-align: middle;
    border-radius: 2px;
    animation: cursorBlink 0.7s ease infinite;
  }
  .ls-subtitle { animation: subtitleFade 0.6s ease both; }
  .ls-fading   { animation: screenFade 0.6s ease forwards; }
`;

export default function LoadingScreen({ onDone }) {
    const FULL_TEXT = "SignVision";
    const [displayed, setDisplayed] = useState("");
    const [showSub, setShowSub] = useState(false);
    const [showCursor, setShowCursor] = useState(true);
    const [fading, setFading] = useState(false);

    useEffect(() => {
        let i = 0;
        const typeInterval = setInterval(() => {
            i++;
            setDisplayed(FULL_TEXT.slice(0, i));
            if (i >= FULL_TEXT.length) {
                clearInterval(typeInterval);
                setTimeout(() => setShowSub(true), 250);
                setTimeout(() => setShowCursor(false), 900);
                setTimeout(() => setFading(true), 1500);
                setTimeout(() => onDone(), 2100);
            }
        }, 90);
        return () => clearInterval(typeInterval);
    }, []);

    return (
        <div className={fading ? "ls-fading" : ""} style={ls.overlay}>
            <style>{CSS}</style>

            {/* Background blobs */}
            <div style={ls.blob1} />
            <div style={ls.blob2} />

            {/* Center */}
            <div style={ls.center}>

                {/* Logo icon — big, animated entrance */}
                <div className="ls-logo-wrap ls-glow" style={{ marginBottom: "24px" }}>
                    <LogoIcon size={96} />
                </div>

                {/* Typewriter text */}
                <div style={ls.textRow}>
                    <span style={ls.typedText}>{displayed}</span>
                    {showCursor && <span className="ls-cursor" />}
                </div>

                {/* Subtitle */}
                {showSub && (
                    <div className="ls-subtitle" style={ls.subtitle}>
                        Sign Language Recognition
                    </div>
                )}

                {/* Dots */}
                <div style={ls.dotsRow}>
                    {[0, 1, 2].map(i => (
                        <div key={i} style={{ ...ls.dot, animation: `dotPulse 1.2s ease infinite`, animationDelay: `${i * 0.25}s` }} />
                    ))}
                </div>
            </div>
        </div>
    );
}

const ls = {
    overlay: {
        position: "fixed", inset: 0, zIndex: 9999,
        background: "#05080f",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexDirection: "column",
        fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
    },
    blob1: {
        position: "absolute", top: "-20%", left: "-20%",
        width: "60vw", height: "60vw", borderRadius: "50%",
        background: "radial-gradient(circle, rgba(16,232,184,0.06) 0%, transparent 70%)",
        pointerEvents: "none",
    },
    blob2: {
        position: "absolute", bottom: "-20%", right: "-20%",
        width: "60vw", height: "60vw", borderRadius: "50%",
        background: "radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%)",
        pointerEvents: "none",
    },
    center: {
        display: "flex", flexDirection: "column",
        alignItems: "center", gap: "14px",
        position: "relative", zIndex: 1,
    },
    textRow: {
        display: "flex", alignItems: "center",
        minHeight: "60px",
    },
    typedText: {
        fontSize: "clamp(2.4rem, 6vw, 4rem)",
        fontWeight: 800, color: "#e8f4ff",
        letterSpacing: "-0.02em", lineHeight: 1,
        fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
    },
    subtitle: {
        fontSize: "0.9rem", color: "rgba(232,244,255,0.45)",
        letterSpacing: "0.1em", textTransform: "uppercase",
        fontFamily: "'Space Mono', monospace",
    },
    dotsRow: {
        display: "flex", gap: "8px", marginTop: "20px",
    },
    dot: {
        width: "8px", height: "8px", borderRadius: "50%",
        background: "#10e8b8",
    },
};