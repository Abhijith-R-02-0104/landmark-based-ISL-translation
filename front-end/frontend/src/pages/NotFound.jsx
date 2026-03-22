import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import LogoIcon from "../components/LogoIcon";

const CSS = `
  @keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-12px); }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes glitch1 {
    0%,100% { clip-path: inset(0 0 95% 0); transform: translate(-4px, 0); }
    20%     { clip-path: inset(30% 0 50% 0); transform: translate(4px, 0); }
    40%     { clip-path: inset(60% 0 20% 0); transform: translate(-4px, 0); }
    60%     { clip-path: inset(80% 0 5% 0);  transform: translate(4px, 0); }
    80%     { clip-path: inset(10% 0 80% 0); transform: translate(-4px, 0); }
  }
  @keyframes glitch2 {
    0%,100% { clip-path: inset(50% 0 30% 0); transform: translate(4px, 0); }
    20%     { clip-path: inset(10% 0 70% 0);  transform: translate(-4px, 0); }
    40%     { clip-path: inset(70% 0 10% 0);  transform: translate(4px, 0); }
    60%     { clip-path: inset(20% 0 60% 0);  transform: translate(-4px, 0); }
    80%     { clip-path: inset(90% 0 2% 0);   transform: translate(4px, 0); }
  }
  @keyframes scanline404 {
    0%   { top: 0%; opacity: 0.06; }
    50%  { opacity: 0.12; }
    100% { top: 100%; opacity: 0.04; }
  }
  @keyframes pulse404 {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,232,184,0.4); }
    50%     { box-shadow: 0 0 0 16px rgba(16,232,184,0); }
  }

  .nf-anim-1 { animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) 0.05s both; }
  .nf-anim-2 { animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) 0.15s both; }
  .nf-anim-3 { animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) 0.25s both; }
  .nf-anim-4 { animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) 0.35s both; }
  .nf-anim-5 { animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) 0.45s both; }

  .nf-logo { animation: floatY 3.5s ease-in-out infinite; }

  .nf-404-wrap { position: relative; display: inline-block; }
  .nf-404-main {
    font-family: 'Space Mono', monospace;
    font-size: clamp(7rem, 18vw, 14rem);
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(135deg, #10e8b8 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    letter-spacing: -4px;
  }
  .nf-404-ghost {
    position: absolute; inset: 0;
    font-family: 'Space Mono', monospace;
    font-size: clamp(7rem, 18vw, 14rem);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -4px;
    color: #10e8b8;
    pointer-events: none;
  }
  .nf-404-ghost.g1 { animation: glitch1 4s infinite steps(1); opacity: 0.4; }
  .nf-404-ghost.g2 { animation: glitch2 4s infinite steps(1); opacity: 0.3; color: #6366f1; }

  .nf-btn {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 14px 32px; border-radius: 14px; border: none;
    background: linear-gradient(135deg, var(--accent), #0abf97);
    color: var(--accent-btn-text); font-weight: 700; font-size: 1rem;
    cursor: pointer; font-family: inherit;
    box-shadow: 0 12px 30px var(--accent-dim);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    animation: pulse404 2.5s ease-in-out infinite;
  }
  .nf-btn:hover { transform: translateY(-3px); box-shadow: 0 18px 40px var(--accent-dim); filter: brightness(1.08); }
  .nf-btn:active { transform: scale(0.96); }

  .nf-ghost-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 24px; border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface); color: var(--text-muted);
    font-weight: 600; font-size: 0.95rem; cursor: pointer;
    font-family: inherit; transition: all 0.22s ease;
  }
  .nf-ghost-btn:hover { border-color: var(--accent-border); background: var(--accent-dim); color: var(--accent); }

  .nf-scanline {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(16,232,184,0.4), transparent);
    animation: scanline404 3s ease-in-out infinite;
    pointer-events: none;
  }

  .nf-links a {
    color: var(--text-muted); text-decoration: none; font-size: 0.88rem;
    transition: color 0.2s ease; padding: 4px 8px; border-radius: 6px;
  }
  .nf-links a:hover { color: var(--accent); background: var(--accent-dim); }
`;

export default function NotFound() {
    const navigate = useNavigate();
    const canvasRef = useRef(null);

    // Mini particle effect
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        let animId;

        const particles = Array.from({ length: 30 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            alpha: Math.random() * 0.3 + 0.05,
        }));

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(16,232,184,${p.alpha})`;
                ctx.fill();
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0) p.x = canvas.width;
                if (p.x > canvas.width) p.x = 0;
                if (p.y < 0) p.y = canvas.height;
                if (p.y > canvas.height) p.y = 0;
            });
            animId = requestAnimationFrame(draw);
        };
        draw();
        return () => cancelAnimationFrame(animId);
    }, []);

    return (
        <div style={s.page}>
            <style>{CSS}</style>

            {/* Particles */}
            <canvas ref={canvasRef} style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none", opacity: 0.5 }} />

            {/* Scanline */}
            <div className="nf-scanline" />

            {/* Content */}
            <div style={s.center}>

                {/* Logo */}
                <div className="nf-anim-1 nf-logo">
                    <LogoIcon size={72} />
                </div>

                {/* 404 with glitch */}
                <div className="nf-anim-2 nf-404-wrap">
                    <div className="nf-404-ghost g1">404</div>
                    <div className="nf-404-ghost g2">404</div>
                    <div className="nf-404-main">404</div>
                </div>

                {/* Message */}
                <div className="nf-anim-3" style={s.title}>
                    Page Not Found
                </div>
                <div className="nf-anim-4" style={s.subtitle}>
                    Looks like this page doesn't exist.<br />
                    Maybe you mistyped the URL or the page was moved.
                </div>

                {/* Buttons */}
                <div className="nf-anim-5" style={s.btnRow}>
                    <button className="nf-btn" onClick={() => navigate("/")}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M19 12H5M12 19l-7-7 7-7" />
                        </svg>
                        Back to Home
                    </button>
                    <button className="nf-ghost-btn" onClick={() => navigate("/detect")}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M23 7l-7 5 7 5V7z" /><rect x="1" y="5" width="15" height="14" rx="2" />
                        </svg>
                        Go to Detection
                    </button>
                </div>

                {/* Quick nav links */}
                <div className="nf-anim-5 nf-links" style={s.links}>
                    <span style={{ color: "var(--text-faint)", fontSize: "0.82rem" }}>Or go to:</span>
                    <a onClick={() => navigate("/")}>Home</a>
                    <a onClick={() => navigate("/detect")}>Detection</a>
                    <a onClick={() => navigate("/about")}>Team</a>
                    <a onClick={() => navigate("/abstract")}>Abstract</a>
                </div>

                {/* Error code */}
                <div className="nf-anim-5" style={s.errorCode}>
                    <span style={{ color: "var(--text-faint)", fontFamily: "'Space Mono',monospace", fontSize: "0.75rem" }}>
                        ERROR_CODE: 404 · NOT_FOUND · SIGNVISION_V1
                    </span>
                </div>
            </div>
        </div>
    );
}

const s = {
    page: {
        minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: "'Plus Jakarta Sans',system-ui,sans-serif", color: "var(--text)",
        position: "relative", overflow: "hidden", padding: "24px",
    },
    center: {
        position: "relative", zIndex: 1,
        display: "flex", flexDirection: "column", alignItems: "center",
        textAlign: "center", gap: "20px", maxWidth: "600px",
    },
    title: {
        fontSize: "clamp(1.6rem, 4vw, 2.4rem)", fontWeight: 800,
        color: "var(--text)", letterSpacing: "-0.02em", margin: 0,
    },
    subtitle: {
        fontSize: "1rem", color: "var(--text-muted)", lineHeight: 1.75,
        maxWidth: "440px", margin: 0,
    },
    btnRow: { display: "flex", gap: "12px", flexWrap: "wrap", justifyContent: "center", marginTop: "8px" },
    links: { display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap", justifyContent: "center", marginTop: "8px" },
    errorCode: { marginTop: "16px", opacity: 0.5 },
};