import { useNavigate } from "react-router-dom";
import { usePageTitle } from "../App";
import LogoIcon from "../components/LogoIcon";

const CSS = `
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmerTitle {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
  }
  @keyframes floatBadge {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-4px); }
  }
  @keyframes logoFloat {
    0%,100% { transform: translateY(0) rotate(-1deg); filter: drop-shadow(0 8px 24px rgba(19,237,212,0.35)); }
    50%     { transform: translateY(-8px) rotate(1deg); filter: drop-shadow(0 16px 36px rgba(19,237,212,0.55)); }
  }
  @keyframes glowPulseGold {
    0%,100% { box-shadow: 0 0 0 0 rgba(212,175,55,0.4); }
    50%     { box-shadow: 0 0 0 10px rgba(212,175,55,0); }
  }

  .anim-1{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .05s both}
  .anim-2{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .15s both}
  .anim-3{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .25s both}
  .anim-4{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .35s both}
  .anim-5{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .45s both}
  .anim-6{animation:fadeInUp .6s cubic-bezier(.22,.68,0,1.2) .55s both}

  .about-shimmer {
    background: linear-gradient(90deg, var(--text) 0%, var(--accent) 40%, var(--purple) 60%, var(--text) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: shimmerTitle 5s linear infinite;
  }

  .sv-logo-float { animation: logoFloat 4s ease-in-out infinite; }

  .member-card {
    position: relative; overflow: hidden;
    transition: transform 0.38s cubic-bezier(.22,.68,0,1.1), border-color 0.38s ease, box-shadow 0.38s ease;
    cursor: default;
  }
  .member-card:hover {
    transform: translateY(-10px);
    border-color: var(--accent-border) !important;
    box-shadow: 0 32px 70px var(--accent-dim), 0 0 0 1px var(--accent-border) !important;
  }
  .member-avatar { transition: all 0.35s ease; }
  .member-card:hover .member-avatar { box-shadow: 0 0 32px var(--accent-dim), 0 0 0 3px var(--accent-border); transform: scale(1.08); }
  .role-tag { transition: all 0.3s ease; }
  .member-card:hover .role-tag { background: var(--accent-dim) !important; border-color: var(--accent-border) !important; color: var(--accent) !important; }
  .card-bar { transition: opacity 0.3s ease; opacity: 0.5; }
  .member-card:hover .card-bar { opacity: 1; }
  .guide-card { transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease; }
  .guide-card:hover { transform: translateY(-4px); border-color: var(--accent-border) !important; box-shadow: 0 20px 50px var(--accent-dim) !important; }
  .stat-chip { transition: all 0.25s ease; }
  .stat-chip:hover { border-color: var(--accent-border) !important; background: var(--accent-dim) !important; }
  .back-btn { transition: all 0.22s ease; }
  .back-btn:hover { border-color: var(--accent-border) !important; background: var(--accent-dim) !important; }
`;

const team = [
    { initials: "AR", name: "Abhijith R", role: "Developer", grad: "linear-gradient(135deg,#10e8b8,#0891b2)" },
    { initials: "JB", name: "Joshua K Benny", role: "Developer", grad: "linear-gradient(135deg,#6366f1,#8b5cf6)" },
    { initials: "NS", name: "Namith S Nair", role: "Developer", grad: "linear-gradient(135deg,#f59e0b,#ef4444)" },
    { initials: "NS", name: "Navin S", role: "Developer", grad: "linear-gradient(135deg,#ec4899,#8b5cf6)" },
];

export default function About() {
    const navigate = useNavigate();
    usePageTitle("Team");

    return (
        <main style={a.page}>
            <style>{CSS}</style>

            {/* ── Hero ──────────────────────────────────────────── */}
            <section style={a.hero}>

                {/* College logo — large, glowing, premium */}
                <div className="anim-1" style={a.logoWrap}>
                    <img
                        src="/college-logo.png"
                        alt="Sree Buddha College of Engineering"
                        className="college-logo-hero"
                        style={a.heroLogo}
                    />
                </div>

                {/* College badge */}
                <div className="anim-1" style={a.collegeBadge}>
                    <span style={a.badgeDot} />
                    Sree Buddha College of Engineering, Pattoor
                </div>

                <h1 className="anim-2 about-shimmer" style={a.heroTitle}>
                    Meet the Team
                </h1>

                {/* Project info box */}
                <div className="anim-3" style={a.projectBox}>
                    <div style={a.projectLabel}>Project</div>
                    <div style={a.projectTitle}>
                        Landmark-Based Dynamic Sign Language Recognition
                        Using Temporal Hand Motion Analysis
                    </div>
                    <div style={a.projectDept}>Dept. of CSE · Autonomous · 2025</div>
                </div>

                {/* Stats */}
                <div className="anim-4" style={a.statsRow}>
                    {[
                        { v: "GRU", l: "Model" },
                        { v: "85–95%", l: "Accuracy" },
                        { v: "30×63", l: "Input Shape" },
                        { v: "2025", l: "Academic Year" },
                    ].map(s => (
                        <div key={s.v} className="stat-chip glass-card" style={a.statChip}>
                            <span style={a.statVal}>{s.v}</span>
                            <span style={a.statLbl}>{s.l}</span>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── Team Cards ────────────────────────────────────── */}
            <section style={a.section}>
                <div className="anim-5" style={a.sectionHead}>
                    <div style={a.sectionPill}>Developers</div>
                    <h2 style={a.sectionTitle}>Development Team</h2>
                </div>
                <div className="anim-6" style={a.teamGrid}>
                    {team.map((m, i) => (
                        <div key={i} className="member-card glass-card micro-card" style={{ ...a.memberCard, position: "relative", overflow: "hidden" }}>
                            <div className="member-avatar" style={{ ...a.avatar, background: m.grad }}>{m.initials}</div>
                            <div style={a.memberName}>{m.name}</div>
                            <div className="role-tag" style={a.roleTag}>{m.role}</div>
                            <div className="card-bar" style={{ ...a.cardBar, background: m.grad }} />
                        </div>
                    ))}
                </div>
            </section>

            {/* ── Guides ────────────────────────────────────────── */}
            <section style={{ ...a.section, paddingTop: 0 }}>
                <div style={a.sectionHead}>
                    <div style={a.sectionPill}>Mentors</div>
                    <h2 style={a.sectionTitle}>Project Guide & Supervisor</h2>
                </div>
                <div style={a.guideWrap}>
                    {/* Ms. Aswathy T */}
                    <div className="guide-card glass-card" style={{ ...a.guideCard, position: "relative", overflow: "hidden" }}>
                        <div style={a.guideBar} />
                        <div style={a.guideAvatar}>AT</div>
                        <div style={a.guideInfo}>
                            <div style={a.guideName}>Ms. Aswathy T</div>
                            <div style={a.guideRole}>Assistant Professor — Department of CSE</div>
                            <div style={a.guideCollege}>Sree Buddha College of Engineering, Pattoor</div>
                        </div>
                        <div style={a.guideBadge}>
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 10v6M2 10l10-5 10 5-10 5z" /><path d="M6 12v5c3 3 9 3 12 0v-5" />
                            </svg>
                            Project Guide
                        </div>
                    </div>
                    {/* Ms. Jyothi B */}
                    <div className="guide-card glass-card" style={{ ...a.guideCard, marginTop: 0, position: "relative", overflow: "hidden" }}>
                        <div style={{ ...a.guideBar, background: "linear-gradient(180deg,#6366f1,#8b5cf6)" }} />
                        <div style={{ ...a.guideAvatar, background: "linear-gradient(135deg,#6366f1,#8b5cf6)" }}>JB</div>
                        <div style={a.guideInfo}>
                            <div style={a.guideName}>Ms. Jyothi B</div>
                            <div style={a.guideRole}>Assistant Professor — Department of CSE</div>
                            <div style={a.guideCollege}>Sree Buddha College of Engineering, Pattoor</div>
                        </div>
                        <div style={{ ...a.guideBadge, background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)", color: "#6366f1" }}>
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 10v6M2 10l10-5 10 5-10 5z" /><path d="M6 12v5c3 3 9 3 12 0v-5" />
                            </svg>
                            Project Supervisor
                        </div>
                    </div>
                </div>
            </section>

            {/* ── College Banner ────────────────────────────────── */}
            <section style={{ ...a.section, paddingTop: 0, paddingBottom: "80px" }}>
                <div style={a.collegeCard}>
                    <div style={a.collegeGlow} />

                    {/* SignVision logo in banner */}
                    <div style={{ flexShrink: 0, position: "relative", zIndex: 1 }}>
                        <LogoIcon size={64} />
                    </div>

                    {/* College logo */}
                    <img
                        src="/college-logo.png"
                        alt="SBCE"
                        style={a.collegeLogo}
                    />

                    <div style={a.collegeContent}>
                        <div style={a.collegeName}>Sree Buddha College of Engineering</div>
                        <div style={a.collegeDetail}>Autonomous · Pattoor, Alappuzha, Kerala · Est. 2002</div>
                        <div style={a.collegeDept}>Department of Computer Science and Engineering</div>
                    </div>

                    <button className="back-btn" style={a.backBtn} onClick={() => navigate("/")}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M19 12H5M12 19l-7-7 7-7" />
                        </svg>
                        Back to Home
                    </button>
                </div>
            </section>
        </main>
    );
}

const a = {
    page: { maxWidth: "1100px", margin: "0 auto", padding: "0 24px 40px", fontFamily: "'Plus Jakarta Sans',system-ui,sans-serif", color: "var(--text)" },

    hero: { textAlign: "center", padding: "48px 24px 44px", display: "flex", flexDirection: "column", alignItems: "center", gap: "18px" },
    logoWrap: { display: "flex", justifyContent: "center", marginBottom: "4px" },
    heroLogo: { width: "110px", height: "110px", objectFit: "contain", borderRadius: "50%", background: "#fff", padding: "4px", filter: "drop-shadow(0 0 18px rgba(212,175,55,0.55))", cursor: "pointer" },

    collegeBadge: { display: "inline-flex", alignItems: "center", gap: "8px", padding: "8px 16px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "13px", color: "var(--accent)", animation: "floatBadge 3s ease infinite" },
    badgeDot: { width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", flexShrink: 0 },

    heroTitle: { fontSize: "clamp(2.8rem,6vw,5rem)", fontWeight: 800, lineHeight: 1.05, margin: "0", letterSpacing: "-0.02em" },

    projectBox: { padding: "20px 28px", borderRadius: "18px", background: "var(--surface)", border: "1px solid var(--border)", maxWidth: "640px", textAlign: "center" },
    projectLabel: { fontSize: "0.72rem", fontWeight: 700, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" },
    projectTitle: { fontSize: "1rem", fontWeight: 700, color: "var(--text)", lineHeight: 1.6, marginBottom: "8px" },
    projectDept: { fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "'Space Mono',monospace" },

    statsRow: { display: "flex", gap: "12px", flexWrap: "wrap", justifyContent: "center" },
    statChip: { display: "inline-flex", flexDirection: "column", alignItems: "center", padding: "14px 22px", borderRadius: "16px", background: "var(--surface)", border: "1px solid var(--border)", cursor: "default", minWidth: "90px" },
    statVal: { fontFamily: "'Space Mono',monospace", fontSize: "1.15rem", fontWeight: 700, color: "var(--accent)", lineHeight: 1 },
    statLbl: { fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px", letterSpacing: "0.04em" },

    section: { maxWidth: "900px", margin: "0 auto", padding: "0 0 56px" },
    sectionHead: { textAlign: "center", marginBottom: "36px" },
    sectionPill: { display: "inline-block", padding: "5px 14px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "11px", color: "var(--accent)", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.07em" },
    sectionTitle: { fontSize: "clamp(1.6rem,3.5vw,2.4rem)", fontWeight: 800, margin: "0 0 10px", letterSpacing: "-0.02em", color: "var(--text)" },

    teamGrid: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "16px" },
    memberCard: { padding: "28px 18px 22px", borderRadius: "22px", background: "var(--surface)", border: "1px solid var(--border)", backdropFilter: "blur(12px)", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", position: "relative", overflow: "hidden", boxShadow: "0 4px 20px var(--shadow)" },
    avatar: { width: "72px", height: "72px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.3rem", fontWeight: 800, color: "#fff", marginBottom: "16px", boxShadow: "0 8px 24px rgba(0,0,0,0.2)", letterSpacing: "0.02em", flexShrink: 0 },
    memberName: { fontSize: "1.05rem", fontWeight: 800, color: "var(--text)", letterSpacing: "-0.01em", marginBottom: "10px" },
    roleTag: { padding: "5px 12px", borderRadius: "999px", background: "var(--surface-2)", border: "1px solid var(--border)", fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "8px" },
    cardBar: { position: "absolute", bottom: 0, left: 0, right: 0, height: "3px" },

    guideWrap: { display: "flex", flexDirection: "column", gap: "14px", alignItems: "stretch", maxWidth: "680px", margin: "0 auto" },
    guideCard: { display: "flex", alignItems: "center", gap: "24px", flexWrap: "wrap", padding: "28px 32px", borderRadius: "22px", background: "var(--surface)", border: "1px solid var(--border)", backdropFilter: "blur(12px)", position: "relative", overflow: "hidden", width: "100%", boxShadow: "0 4px 24px var(--shadow)" },
    guideBar: { position: "absolute", left: 0, top: 0, bottom: 0, width: "4px", background: "linear-gradient(180deg,var(--accent),var(--purple))" },
    guideAvatar: { width: "64px", height: "64px", borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(135deg,var(--accent),var(--purple))", fontSize: "1.1rem", fontWeight: 800, color: "var(--accent-btn-text)", boxShadow: "0 8px 24px var(--accent-dim)" },
    guideInfo: { flex: 1 },
    guideName: { fontSize: "1.2rem", fontWeight: 800, color: "var(--text)", marginBottom: "4px", letterSpacing: "-0.01em" },
    guideRole: { fontSize: "0.88rem", color: "var(--text-muted)", marginBottom: "4px" },
    guideCollege: { fontSize: "0.78rem", color: "var(--text-faint)", fontFamily: "'Space Mono',monospace" },
    guideBadge: { display: "inline-flex", alignItems: "center", gap: "7px", flexShrink: 0, padding: "8px 14px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "0.75rem", fontWeight: 700, color: "var(--accent)", whiteSpace: "nowrap" },

    collegeCard: { position: "relative", overflow: "hidden", padding: "28px 32px", borderRadius: "24px", background: "var(--surface)", border: "1px solid var(--border)", display: "flex", alignItems: "center", gap: "20px", flexWrap: "wrap", boxShadow: "0 4px 24px var(--shadow)" },
    collegeGlow: { position: "absolute", inset: 0, pointerEvents: "none", background: "radial-gradient(ellipse at 0% 50%, rgba(19,237,212,0.06) 0%, transparent 60%)" },
    collegeLogo: { width: "60px", height: "60px", objectFit: "contain", borderRadius: "50%", background: "#fff", padding: "3px", flexShrink: 0, position: "relative", zIndex: 1, filter: "drop-shadow(0 0 12px rgba(212,175,55,0.5))", animation: "glowPulseGold 3s ease-in-out infinite" },
    collegeContent: { flex: 1, position: "relative", zIndex: 1 },
    collegeName: { fontSize: "1.1rem", fontWeight: 800, color: "var(--text)", marginBottom: "4px", letterSpacing: "-0.01em" },
    collegeDetail: { fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: "3px" },
    collegeDept: { fontSize: "0.78rem", color: "var(--accent)", fontFamily: "'Space Mono',monospace" },
    backBtn: { display: "inline-flex", alignItems: "center", gap: "8px", position: "relative", zIndex: 1, padding: "11px 20px", borderRadius: "12px", border: "1px solid var(--accent-border)", background: "var(--accent-dim)", color: "var(--accent)", fontWeight: 700, fontSize: "0.88rem", fontFamily: "inherit", cursor: "pointer" },
};