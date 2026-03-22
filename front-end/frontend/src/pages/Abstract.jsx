import { useNavigate } from "react-router-dom";
import { usePageTitle } from "../App";

const CSS = `
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmerTitle {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
  }
  .anim-1{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .05s both}
  .anim-2{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .15s both}
  .anim-3{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .25s both}
  .anim-4{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .35s both}
  .anim-5{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .45s both}
  .anim-6{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .55s both}
  .anim-7{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .65s both}
  .anim-8{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .75s both}

  .abs-shimmer {
    background: linear-gradient(90deg, var(--text) 0%, var(--accent) 40%, var(--purple) 60%, var(--text) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: shimmerTitle 5s linear infinite;
  }

  .abs-section-card {
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  }
  .abs-section-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-border) !important;
    box-shadow: 0 20px 50px var(--accent-dim) !important;
  }

  .obj-item { transition: all 0.25s ease; }
  .obj-item:hover { background: var(--accent-dim) !important; border-color: var(--accent-border) !important; }

  .ref-chip { transition: all 0.22s ease; }
  .ref-chip:hover { background: var(--accent-dim) !important; border-color: var(--accent-border) !important; color: var(--accent) !important; }

  .back-btn-abs { transition: all 0.22s ease; }
  .back-btn-abs:hover { border-color: var(--accent-border) !important; background: var(--accent-dim) !important; }
`;

const objectives = [
    "Develop a real-time sign language recognition system using hand landmark detection",
    "Implement a GRU-based deep learning model for temporal gesture classification",
    "Extract and process 21 3D hand landmarks per frame using MediaPipe",
    "Build a full-stack application with FastAPI backend and React frontend",
    "Achieve high accuracy recognition of dynamic sign language gestures",
];

const methodology = [
    { step: "01", title: "Data Collection", desc: "Custom dataset collected using MediaPipe landmark extraction across multiple signers and lighting conditions." },
    { step: "02", title: "Preprocessing", desc: "63 landmark coordinates (x, y, z × 21) normalized and structured into 30-frame temporal sequences." },
    { step: "03", title: "Model Architecture", desc: "GRU (Gated Recurrent Unit) network trained on sequential landmark data to capture temporal motion patterns." },
    { step: "04", title: "Backend Integration", desc: "FastAPI server exposes a /predict endpoint accepting base64 frames, returning letter, confidence, and current word." },
    { step: "05", title: "Frontend Interface", desc: "React + Vite frontend with real-time webcam feed, AR overlay, confidence visualization, and sentence building." },
];

const results = [
    { label: "Validation Accuracy", value: "85 – 95%", color: "var(--accent)" },
    { label: "Input Shape", value: "30 × 63", color: "#6366f1" },
    { label: "Landmarks per Hand", value: "21 (3D)", color: "#f59e0b" },
    { label: "Frame Rate", value: "Real-time", color: "#10b981" },
    { label: "Model Type", value: "GRU/Keras", color: "var(--accent)" },
    { label: "Backbone", value: "MediaPipe", color: "#f59e0b" },
];

const futureScope = [
    { icon: "🌐", title: "Multi-language Support", desc: "Extend recognition to support ISL (Indian Sign Language) and ASL simultaneously with language switching." },
    { icon: "🤲", title: "Two-hand Gesture Support", desc: "Current model supports single-hand detection. Future work includes dual-hand complex gesture recognition." },
    { icon: "🧠", title: "Transformer Architecture", desc: "Replace GRU with transformer-based attention models for improved long-sequence gesture understanding." },
    { icon: "📱", title: "Mobile Deployment", desc: "Optimize the model for on-device inference using TensorFlow Lite for mobile and edge deployment." },
    { icon: "🔊", title: "Voice + Sign Integration", desc: "Combine speech recognition with sign language output for a fully bidirectional communication system." },
];

const references = [
    "MediaPipe Hands — Google Research, 2020",
    "TensorFlow / Keras — Abadi et al., 2016",
    "Gated Recurrent Units — Cho et al., 2014",
    "FastAPI — Ramírez, S., 2018",
    "React — Meta Open Source, 2013",
    "OpenCV — Bradski, G., 2000",
];

export default function Abstract() {
    const navigate = useNavigate();
    usePageTitle("Abstract");

    return (
        <main style={p.page}>
            <style>{CSS}</style>

            {/* ── Header ─────────────────────────────── */}
            <div style={p.topBar}>
                <button className="back-btn-abs" style={p.backBtn} onClick={() => navigate("/")}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
                    Back to Home
                </button>
                <div style={p.headerBadge}>Academic Document</div>
            </div>

            {/* ── Hero ───────────────────────────────── */}
            <section style={p.hero}>
                <div className="anim-1" style={p.journalPill}>
                    <span style={p.pillDot} />
                    B.Tech Mini Project · 2023–27
                </div>
                <h1 className="anim-2 abs-shimmer" style={p.title}>Project Abstract</h1>
                <div className="anim-3" style={p.titleCard}>
                    <div style={p.titleLabel}>Full Title</div>
                    <div style={p.titleText}>
                        Landmark-Based Dynamic Sign Language Recognition Using Temporal Hand Motion Analysis
                    </div>
                    <div style={p.titleMeta}>
                        <span style={p.metaChip}>Dept. of CSE</span>
                        <span style={p.metaChip}>Sree Buddha College of Engineering</span>
                        <span style={p.metaChip}>Autonomous · Pattoor, Kerala</span>
                        <span style={p.metaChip}>Academic Year 2023–27</span>
                    </div>
                </div>
            </section>

            {/* ── Abstract ───────────────────────────── */}
            <section style={p.section}>
                <div className="anim-4 abs-section-card glass-card" style={{ ...p.card, borderLeft: "4px solid var(--accent)" }}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "var(--accent-dim)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>01</div>
                        <h2 style={p.cardTitle}>Abstract</h2>
                    </div>
                    <p style={p.para}>
                        Sign language is the primary mode of communication for the hearing-impaired community.
                        This project presents a real-time dynamic sign language recognition system that leverages
                        MediaPipe's hand landmark detection framework to extract 21 three-dimensional keypoints
                        per frame, forming temporal sequences of 30 frames as input to a Gated Recurrent Unit
                        (GRU) deep learning model.
                    </p>
                    <p style={p.para}>
                        The system achieves 85–95% validation accuracy and operates in real-time through a
                        FastAPI backend integrated with a React-based frontend interface. The pipeline covers
                        video capture, landmark extraction, feature engineering, sequence modeling, and
                        sign-to-text conversion — enabling fluid, accessible communication for hearing-impaired individuals.
                    </p>
                </div>
            </section>

            {/* ── Objectives ─────────────────────────── */}
            <section style={p.section}>
                <div className="anim-5 abs-section-card glass-card" style={p.card}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "rgba(99,102,241,0.12)", color: "#6366f1", border: "1px solid rgba(99,102,241,0.25)" }}>02</div>
                        <h2 style={p.cardTitle}>Objectives</h2>
                    </div>
                    <div style={p.objGrid}>
                        {objectives.map((obj, i) => (
                            <div key={i} className="obj-item" style={p.objItem}>
                                <div style={p.objNum}>{String(i + 1).padStart(2, "0")}</div>
                                <div style={p.objText}>{obj}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Methodology ────────────────────────── */}
            <section style={p.section}>
                <div className="anim-6 abs-section-card glass-card" style={p.card}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "rgba(245,158,11,0.12)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.25)" }}>03</div>
                        <h2 style={p.cardTitle}>Methodology Overview</h2>
                    </div>
                    <div style={p.methodGrid}>
                        {methodology.map((m, i) => (
                            <div key={i} style={p.methodItem}>
                                <div style={p.methodStep}>{m.step}</div>
                                <div style={p.methodLine} />
                                <div style={p.methodContent}>
                                    <div style={p.methodTitle}>{m.title}</div>
                                    <div style={p.methodDesc}>{m.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Results ────────────────────────────── */}
            <section style={p.section}>
                <div className="anim-7 abs-section-card glass-card" style={p.card}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "rgba(16,185,129,0.12)", color: "#10b981", border: "1px solid rgba(16,185,129,0.25)" }}>04</div>
                        <h2 style={p.cardTitle}>Results & Accuracy</h2>
                    </div>
                    <div style={p.resultsGrid}>
                        {results.map((r, i) => (
                            <div key={i} style={p.resultCard}>
                                <div style={{ ...p.resultValue, color: r.color }}>{r.value}</div>
                                <div style={p.resultLabel}>{r.label}</div>
                            </div>
                        ))}
                    </div>
                    <div style={p.resultNote}>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" /></svg>
                        Model trained on custom-collected dataset. Accuracy varies with lighting conditions, hand orientation, and signer variability.
                    </div>
                </div>
            </section>

            {/* ── Future Scope ───────────────────────── */}
            <section style={p.section}>
                <div className="anim-8 abs-section-card glass-card" style={p.card}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "rgba(236,72,153,0.12)", color: "#ec4899", border: "1px solid rgba(236,72,153,0.25)" }}>05</div>
                        <h2 style={p.cardTitle}>Future Scope</h2>
                    </div>
                    <div style={p.futureGrid}>
                        {futureScope.map((f, i) => (
                            <div key={i} style={p.futureCard}>
                                <div style={p.futureIcon}>{f.icon}</div>
                                <div style={p.futureTitle}>{f.title}</div>
                                <div style={p.futureDesc}>{f.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── References ─────────────────────────── */}
            <section style={{ ...p.section, paddingBottom: "80px" }}>
                <div style={p.card}>
                    <div style={p.cardHeader}>
                        <div style={{ ...p.sectionNum, background: "var(--surface-2)", color: "var(--text-muted)", border: "1px solid var(--border)" }}>06</div>
                        <h2 style={p.cardTitle}>References & Technologies</h2>
                    </div>
                    <div style={p.refGrid}>
                        {references.map((r, i) => (
                            <div key={i} className="ref-chip" style={p.refChip}>
                                <span style={p.refNum}>{String(i + 1).padStart(2, "0")}</span>
                                {r}
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        </main>
    );
}

const p = {
    page: { maxWidth: "900px", margin: "0 auto", padding: "0 24px 40px", fontFamily: "'Plus Jakarta Sans',system-ui,sans-serif", color: "var(--text)" },

    topBar: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 0 32px" },
    backBtn: { display: "inline-flex", alignItems: "center", gap: "7px", background: "none", border: "1px solid var(--border)", color: "var(--text-muted)", cursor: "pointer", fontSize: "0.85rem", padding: "8px 14px", borderRadius: "10px", fontFamily: "inherit" },
    headerBadge: { padding: "6px 14px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "11px", color: "var(--accent)", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" },

    hero: { textAlign: "center", padding: "0 0 48px", display: "flex", flexDirection: "column", alignItems: "center", gap: "20px" },
    journalPill: { display: "inline-flex", alignItems: "center", gap: "8px", padding: "8px 16px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "13px", color: "var(--accent)" },
    pillDot: { width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", flexShrink: 0 },
    title: { fontSize: "clamp(2.4rem,5vw,4rem)", fontWeight: 800, margin: 0, letterSpacing: "-0.02em", lineHeight: 1.1 },

    titleCard: { padding: "24px 28px", borderRadius: "20px", background: "var(--surface)", border: "1px solid var(--border)", maxWidth: "720px", textAlign: "center" },
    titleLabel: { fontSize: "0.72rem", fontWeight: 700, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "10px" },
    titleText: { fontSize: "1.05rem", fontWeight: 700, color: "var(--text)", lineHeight: 1.6, marginBottom: "14px" },
    titleMeta: { display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "center" },
    metaChip: { padding: "4px 12px", borderRadius: "999px", background: "var(--surface-2)", border: "1px solid var(--border)", fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "'Space Mono',monospace" },

    section: { marginBottom: "20px" },
    card: { padding: "28px 32px", borderRadius: "22px", background: "var(--surface)", border: "1px solid var(--border)", backdropFilter: "blur(12px)", boxShadow: "0 4px 24px var(--shadow)" },
    cardHeader: { display: "flex", alignItems: "center", gap: "14px", marginBottom: "20px" },
    sectionNum: { width: "36px", height: "36px", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.75rem", fontWeight: 800, fontFamily: "'Space Mono',monospace", flexShrink: 0 },
    cardTitle: { fontSize: "1.25rem", fontWeight: 800, margin: 0, letterSpacing: "-0.01em", color: "var(--text)" },

    para: { fontSize: "0.95rem", lineHeight: 1.85, color: "var(--text-muted)", margin: "0 0 14px 0" },

    objGrid: { display: "flex", flexDirection: "column", gap: "10px" },
    objItem: { display: "flex", alignItems: "flex-start", gap: "14px", padding: "12px 16px", borderRadius: "12px", background: "var(--surface-2)", border: "1px solid var(--border)" },
    objNum: { fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", fontWeight: 700, color: "var(--accent)", flexShrink: 0, marginTop: "2px" },
    objText: { fontSize: "0.9rem", color: "var(--text-muted)", lineHeight: 1.6 },

    methodGrid: { display: "flex", flexDirection: "column", gap: "0" },
    methodItem: { display: "flex", alignItems: "stretch", gap: "0" },
    methodStep: { width: "40px", height: "40px", borderRadius: "50%", background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.25)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.72rem", fontWeight: 800, color: "#f59e0b", fontFamily: "'Space Mono',monospace", flexShrink: 0, marginTop: "2px" },
    methodLine: { width: "1px", background: "var(--border)", margin: "0 18px", alignSelf: "stretch", minHeight: "48px" },
    methodContent: { paddingBottom: "24px", flex: 1 },
    methodTitle: { fontSize: "0.95rem", fontWeight: 700, color: "var(--text)", marginBottom: "4px" },
    methodDesc: { fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.65 },

    resultsGrid: { display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "14px", marginBottom: "16px" },
    resultCard: { padding: "20px 16px", borderRadius: "16px", background: "var(--surface-2)", border: "1px solid var(--border)", textAlign: "center" },
    resultValue: { fontFamily: "'Space Mono',monospace", fontSize: "1.3rem", fontWeight: 800, lineHeight: 1, marginBottom: "6px" },
    resultLabel: { fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.4 },
    resultNote: { display: "flex", alignItems: "flex-start", gap: "8px", padding: "12px 16px", borderRadius: "12px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "0.82rem", color: "var(--text-muted)", lineHeight: 1.6 },

    futureGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: "14px" },
    futureCard: { padding: "20px", borderRadius: "16px", background: "var(--surface-2)", border: "1px solid var(--border)" },
    futureIcon: { fontSize: "1.5rem", marginBottom: "10px" },
    futureTitle: { fontSize: "0.95rem", fontWeight: 700, color: "var(--text)", marginBottom: "6px" },
    futureDesc: { fontSize: "0.82rem", color: "var(--text-muted)", lineHeight: 1.65 },

    refGrid: { display: "flex", flexDirection: "column", gap: "8px" },
    refChip: { display: "flex", alignItems: "center", gap: "12px", padding: "10px 16px", borderRadius: "10px", background: "var(--surface-2)", border: "1px solid var(--border)", fontSize: "0.85rem", color: "var(--text-muted)", cursor: "default" },
    refNum: { fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", color: "var(--accent)", fontWeight: 700, flexShrink: 0 },
};