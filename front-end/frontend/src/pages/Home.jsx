import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { usePageTitle } from "../App";
import LogoIcon from "../components/LogoIcon";
import Particles from "../components/Particles";

const ANIMATIONS = `
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmerText {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
  }
  @keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-8px); }
  }

  .anim-1{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .05s both}
  .anim-2{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .15s both}
  .anim-3{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .25s both}
  .anim-4{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .35s both}
  .anim-5{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .45s both}
  .anim-6{animation:fadeInUp .65s cubic-bezier(.22,.68,0,1.2) .55s both}

  .shimmer-title {
    background: linear-gradient(90deg, var(--text) 0%, var(--accent) 40%, var(--purple) 60%, var(--text) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: shimmerText 5s linear infinite;
  }

  .stat-card { transition: all 0.35s ease; }
  .stat-card:hover { transform:translateY(-4px); border-color:var(--accent-border) !important; box-shadow:0 20px 50px var(--accent-dim) !important; }

  .step-card { transition: all 0.35s ease; }
  .step-card:hover { transform:translateX(4px); border-color:var(--accent-border) !important; background:var(--accent-dim) !important; }

  .chip-tech { transition: all 0.25s ease; }
  .chip-tech:hover { background:var(--accent-dim) !important; border-color:var(--accent-border) !important; color:var(--accent) !important; }

  .btn-primary { transition: all 0.25s ease; }
  .btn-primary:hover { transform:translateY(-2px); box-shadow:0 18px 40px var(--accent-dim) !important; filter:brightness(1.08); }
  .btn-primary:active { transform:scale(0.97); }

  .btn-secondary { transition: all 0.25s ease; }
  .btn-secondary:hover { transform:translateY(-2px); border-color:var(--accent-border) !important; background:var(--accent-dim) !important; }
  .btn-secondary:active { transform:scale(0.97); }

  .footer-member { transition: color 0.2s ease; }
  .footer-member:hover { color: var(--accent) !important; }

  .footer-link { transition: color 0.2s ease; text-decoration: none; }
  .footer-link:hover { color: var(--accent) !important; }
`;

const steps = [
    { n: "01", title: "Video Capture", text: "Live webcam stream captured via OpenCV at real-time frame rate." },
    { n: "02", title: "Hand Landmark Detection", text: "MediaPipe detects 21 3D landmarks per hand with sub-pixel accuracy." },
    { n: "03", title: "Feature Extraction", text: "63 landmark coordinates (x,y,z) extracted per frame as feature vectors." },
    { n: "04", title: "Temporal Sequence", text: "30-frame sequences formed to capture motion patterns over time." },
    { n: "05", title: "GRU Classification", text: "Gated Recurrent Unit model classifies dynamic sign language words." },
    { n: "06", title: "Sign-to-Text Output", text: "Predicted gesture displayed with confidence score and sentence building." },
];

const techStack = [
    { label: "React", color: "#61dafb" },
    { label: "FastAPI", color: "#10e8b8" },
    { label: "MediaPipe", color: "#ff6d00" },
    { label: "OpenCV", color: "#5c9e31" },
    { label: "TensorFlow", color: "#ff6f00" },
    { label: "GRU / Keras", color: "#6366f1" },
    { label: "Python 3.9+", color: "#ffd43b" },
];

const teamNames = ["Abhijith R", "Joshua K Benny", "Namith S Nair", "Navin S"];

export default function Home() {
    const navigate = useNavigate();
    usePageTitle("Home");
    const scrollToHow = () => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });

    return (
        <main style={s.page}>
            <style>{ANIMATIONS}</style>
            <div style={s.blob1} /><div style={s.blob2} />
            <Particles />

            {/* ── HERO ──────────────────────────────────────────── */}
            <section style={s.hero}>
                <div className="anim-1" style={s.pill}>
                    <span style={s.pillDot} />
                    Landmark-Based Dynamic Sign Language Recognition
                </div>
                <h1 className="anim-2 shimmer-title" style={s.title}>
                    Bridging Communication<br />Through AI Vision
                </h1>
                <p className="anim-3" style={s.subtitle}>
                    Real-time dynamic sign language recognition using MediaPipe landmarks,
                    temporal motion analysis, and a GRU deep learning model — built for
                    hearing-impaired communication.
                </p>
                <div className="anim-4" style={s.btnRow}>
                    <button className="btn-primary micro-btn" style={s.btnPrimary} onClick={() => navigate("/detect")}>
                        <span>Launch Detection</span>
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
                    </button>
                    <button className="btn-secondary micro-btn" style={s.btnSecondary} onClick={scrollToHow}>How It Works</button>
                    <button className="btn-secondary micro-btn" style={s.btnSecondary} onClick={() => navigate("/about")}>Meet the Team</button>
                </div>
                <div className="anim-5" style={s.statsGrid}>
                    {[
                        { value: "21", unit: "landmarks", label: "tracked per hand" },
                        { value: "85–95%", unit: "", label: "validation accuracy" },
                        { value: "30", unit: "frames", label: "per gesture sequence" },
                    ].map((stat, i) => (
                        <div key={stat.value} className="stat-card glass-card micro-card stat-pop" style={{ ...s.statCard, position: "relative", overflow: "hidden", animationDelay: `${i * 0.1}s` }}>
                            <div className="number-roll" style={s.statValue}>{stat.value} <span style={s.statUnit}>{stat.unit}</span></div>
                            <div style={s.statLabel}>{stat.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── HOW IT WORKS ──────────────────────────────────── */}
            <section id="how-it-works" style={s.section}>
                <div style={s.sectionHead}>
                    <div className="pill-float" style={s.sectionPill}>System Pipeline</div>
                    <h2 style={s.sectionTitle}>How It Works</h2>
                    <p style={s.sectionSub}>A 6-stage pipeline from raw webcam input to sign-to-text output.</p>
                </div>
                <div style={s.stepsGrid}>
                    {steps.map(step => (
                        <div key={step.n} className="step-card glass-card micro-card" style={{ ...s.stepCard, position: "relative", overflow: "hidden" }}>
                            <div style={s.stepNum}>{step.n}</div>
                            <div>
                                <div style={s.stepTitle}>{step.title}</div>
                                <div style={s.stepText}>{step.text}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── TECH STACK ────────────────────────────────────── */}
            <section style={s.section}>
                <div style={s.sectionHead}>
                    <div className="pill-float" style={s.sectionPill}>Stack</div>
                    <h2 style={s.sectionTitle}>Technologies Used</h2>
                </div>
                <div style={s.chipGrid}>
                    {techStack.map(t => (
                        <div key={t.label} className="chip-tech micro-btn" style={{ ...s.techChip, borderColor: `${t.color}30` }}>
                            <span style={{ ...s.chipDot, background: t.color }} />
                            {t.label}
                        </div>
                    ))}
                </div>
            </section>

            {/* ── CTA ───────────────────────────────────────────── */}
            <section style={s.cta}>
                <h2 style={s.ctaTitle}>Ready to Detect Sign Language?</h2>
                <p style={s.ctaSub}>Open the detection console and start recognizing gestures in real time.</p>
                <button className="btn-primary micro-btn" style={{ ...s.btnPrimary, padding: "16px 36px", fontSize: "1rem" }} onClick={() => navigate("/detect")}>
                    <span>Open Detection Console</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
                </button>
            </section>

            {/* ── FOOTER ────────────────────────────────────────── */}
            <footer style={s.footer}>
                <div style={s.footerTop}>
                    {/* Left — brand */}
                    <div style={s.footerBrand}>
                        <LogoIcon size={40} />
                        <div>
                            <div style={s.footerName}>SignVision</div>
                            <div style={s.footerTagline}>Sign Language Recognition System</div>
                        </div>
                    </div>

                    {/* Middle — college */}
                    <div style={s.footerMid}>
                        <div style={s.footerCollegeName}>Sree Buddha College of Engineering</div>
                        <div style={s.footerCollegeSub}>Autonomous · Pattoor, Alappuzha, Kerala</div>
                        <div style={s.footerDept}>Dept. of Computer Science and Engineering</div>
                    </div>

                    {/* Right — nav links */}
                    <div style={s.footerNav}>
                        <div style={s.footerNavTitle}>Navigation</div>
                        {[
                            { label: "Home", path: "/" },
                            { label: "Detection", path: "/detect" },
                            { label: "Team", path: "/about" },
                        ].map(link => (
                            <span
                                key={link.label}
                                className="footer-link"
                                style={s.footerNavLink}
                                onClick={() => navigate(link.path)}
                            >
                                {link.label}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Divider */}
                <div style={s.footerDivider} />

                {/* Bottom — team + year */}
                <div style={s.footerBottom}>
                    <div style={s.footerTeamRow}>
                        <span style={{ ...s.footerTeamLabel, color: "var(--text-faint)" }}>Developed by</span>
                        {teamNames.map((name, i) => (
                            <span key={name} className="footer-member" style={s.footerMemberName}>
                                {name}{i < teamNames.length - 1 ? " ·" : ""}
                            </span>
                        ))}
                    </div>
                    <div style={s.footerYear}>© 2025 SignVision · All rights reserved</div>
                </div>
            </footer>
        </main>
    );
}

const s = {
    page: { position: "relative", minHeight: "100vh", overflowX: "hidden", fontFamily: "'Plus Jakarta Sans',system-ui,sans-serif", color: "var(--text)" },
    blob1: { position: "fixed", top: "-200px", left: "-200px", width: "600px", height: "600px", borderRadius: "50%", background: "radial-gradient(circle,var(--accent-dim) 0%,transparent 70%)", pointerEvents: "none", zIndex: 0 },
    blob2: { position: "fixed", bottom: "-200px", right: "-200px", width: "700px", height: "700px", borderRadius: "50%", background: "radial-gradient(circle,rgba(99,102,241,0.07) 0%,transparent 70%)", pointerEvents: "none", zIndex: 0 },

    hero: { position: "relative", zIndex: 1, display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", padding: "72px 24px 48px", maxWidth: "960px", margin: "0 auto" },
    pill: { display: "inline-flex", alignItems: "center", gap: "8px", padding: "8px 16px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "13px", color: "var(--accent)", marginBottom: "28px" },
    pillDot: { width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "floatY 2s ease infinite" },
    title: { fontSize: "clamp(2.8rem,6vw,5.2rem)", fontWeight: 800, lineHeight: 1.05, margin: "0 0 24px", letterSpacing: "-0.02em" },
    subtitle: { fontSize: "1.08rem", lineHeight: 1.75, color: "var(--text-muted)", maxWidth: "680px", margin: "0 0 36px" },
    btnRow: { display: "flex", gap: "12px", flexWrap: "wrap", justifyContent: "center", marginBottom: "52px" },
    btnPrimary: { display: "inline-flex", alignItems: "center", gap: "10px", padding: "14px 28px", borderRadius: "14px", border: "none", background: "linear-gradient(135deg,var(--accent),#0abf97)", color: "var(--accent-btn-text)", fontWeight: 700, fontSize: "0.95rem", cursor: "pointer", fontFamily: "inherit", boxShadow: "0 12px 30px var(--accent-dim)" },
    btnSecondary: { display: "inline-flex", alignItems: "center", gap: "10px", padding: "14px 28px", borderRadius: "14px", border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontWeight: 600, fontSize: "0.95rem", cursor: "pointer", fontFamily: "inherit" },
    statsGrid: { display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "16px", width: "100%", maxWidth: "720px" },
    statCard: { padding: "24px 20px", borderRadius: "20px", textAlign: "center", background: "var(--surface)", border: "1px solid var(--border)", backdropFilter: "blur(12px)" },
    statValue: { fontSize: "2rem", fontWeight: 800, color: "var(--accent)", lineHeight: 1, fontFamily: "'Space Mono',monospace" },
    statUnit: { fontSize: "0.9rem", fontWeight: 400, color: "var(--accent)" },
    statLabel: { marginTop: "8px", fontSize: "0.85rem", color: "var(--text-muted)" },

    section: { position: "relative", zIndex: 1, maxWidth: "1100px", margin: "0 auto", padding: "64px 24px" },
    sectionHead: { textAlign: "center", marginBottom: "48px" },
    sectionPill: { display: "inline-block", padding: "6px 14px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "12px", color: "var(--accent)", marginBottom: "14px", textTransform: "uppercase", letterSpacing: "0.06em" },
    sectionTitle: { fontSize: "clamp(1.8rem,4vw,2.8rem)", fontWeight: 800, margin: "0 0 12px", letterSpacing: "-0.02em", color: "var(--text)" },
    sectionSub: { fontSize: "1rem", color: "var(--text-muted)", maxWidth: "520px", margin: "0 auto" },
    stepsGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: "14px" },
    stepCard: { display: "flex", alignItems: "flex-start", gap: "18px", padding: "20px 22px", borderRadius: "18px", background: "var(--surface)", border: "1px solid var(--border)", cursor: "default" },
    stepNum: { flexShrink: 0, width: "44px", height: "44px", borderRadius: "12px", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontFamily: "'Space Mono',monospace", fontSize: "13px", fontWeight: 700, color: "var(--accent)" },
    stepTitle: { fontWeight: 700, fontSize: "0.97rem", marginBottom: "5px", color: "var(--text)" },
    stepText: { fontSize: "0.88rem", color: "var(--text-muted)", lineHeight: 1.6 },
    chipGrid: { display: "flex", flexWrap: "wrap", gap: "12px", justifyContent: "center" },
    techChip: { display: "inline-flex", alignItems: "center", gap: "8px", padding: "10px 18px", borderRadius: "999px", background: "var(--surface)", border: "1px solid transparent", fontSize: "0.9rem", cursor: "default", color: "var(--text-muted)" },
    chipDot: { width: "8px", height: "8px", borderRadius: "50%", flexShrink: 0 },
    cta: { position: "relative", zIndex: 1, textAlign: "center", padding: "60px 24px 80px", maxWidth: "640px", margin: "0 auto" },
    ctaTitle: { fontSize: "clamp(1.8rem,4vw,2.8rem)", fontWeight: 800, margin: "0 0 16px", letterSpacing: "-0.02em", color: "var(--text)" },
    ctaSub: { fontSize: "1rem", color: "var(--text-muted)", marginBottom: "36px", lineHeight: 1.7 },

    // ── Footer ────────────────────────────────────────────────
    footer: {
        position: "relative", zIndex: 1,
        borderTop: "1px solid var(--border)",
        padding: "48px 40px 32px",
        marginTop: "20px",
        background: "var(--surface)",
        backdropFilter: "blur(12px)",
    },
    footerTop: {
        display: "grid",
        gridTemplateColumns: "1.5fr 2fr 1fr",
        gap: "40px", alignItems: "start",
        marginBottom: "36px",
    },
    footerBrand: { display: "flex", alignItems: "center", gap: "12px" },
    footerLogo: {
        width: "40px", height: "40px", borderRadius: "11px", flexShrink: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "linear-gradient(135deg,var(--accent),#0891b2)",
        fontWeight: 800, fontSize: "12px", color: "var(--accent-btn-text)",
        fontFamily: "'Space Mono',monospace",
        boxShadow: "0 4px 14px var(--accent-dim)",
    },
    footerName: { fontSize: "1rem", fontWeight: 800, color: "var(--text)", letterSpacing: "-0.01em" },
    footerTagline: { fontSize: "0.75rem", color: "var(--text-faint)", marginTop: "2px" },

    footerMid: {},
    footerCollegeName: { fontSize: "0.95rem", fontWeight: 700, color: "var(--text)", marginBottom: "4px" },
    footerCollegeSub: { fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "3px" },
    footerDept: { fontSize: "0.78rem", color: "var(--accent)", fontFamily: "'Space Mono',monospace" },

    footerNav: { display: "flex", flexDirection: "column", gap: "8px" },
    footerNavTitle: { fontSize: "0.72rem", fontWeight: 700, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "4px" },
    footerNavLink: { fontSize: "0.88rem", color: "var(--text-muted)", cursor: "pointer" },

    footerDivider: { height: "1px", background: "var(--border)", marginBottom: "24px" },
    footerBottom: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" },
    footerTeamRow: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" },
    footerTeamLabel: { fontSize: "0.8rem" },
    footerMemberName: { fontSize: "0.82rem", fontWeight: 600, color: "var(--text-muted)", cursor: "default" },
    footerYear: { fontSize: "0.78rem", color: "var(--text-faint)", fontFamily: "'Space Mono',monospace" },
};