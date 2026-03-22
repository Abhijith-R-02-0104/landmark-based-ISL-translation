import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import Home from "./pages/Home";
import Detect from "./pages/Detect";
import About from "./pages/About";
import Abstract from "./pages/Abstract";
import NotFound from "./pages/NotFound";
import LoadingScreen from "./components/LoadingScreen";
import SkeletonLoader from "./components/SkeletonLoader";
import { ToastProvider } from "./components/Toast";
import LogoIcon from "./components/LogoIcon";
import { checkBackendStatus } from "./services/api";

export function usePageTitle(title) {
  useEffect(() => {
    document.title = `SignVision | ${title}`;
    return () => { document.title = "SignVision"; };
  }, [title]);
}

// ── Scroll Progress Bar ────────────────────────────────────
function ScrollProgressBar() {
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    const onScroll = () => {
      const el = document.documentElement;
      const scrolled = el.scrollTop || document.body.scrollTop;
      const total = el.scrollHeight - el.clientHeight;
      setProgress(total > 0 ? (scrolled / total) * 100 : 0);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return (
    <div style={{ position: "fixed", top: 0, left: 0, right: 0, height: "3px", zIndex: 9999, background: "transparent", pointerEvents: "none" }}>
      <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg,var(--accent),#6366f1)", borderRadius: "0 3px 3px 0", transition: "width 0.08s linear", boxShadow: "0 0 8px rgba(16,232,184,0.5)" }} />
    </div>
  );
}

// ── Custom Cursor ──────────────────────────────────────────
function CustomCursor() {
  const dotRef = useRef(null);
  const ringRef = useRef(null);
  const pos = useRef({ x: 0, y: 0 });
  const ring = useRef({ x: 0, y: 0 });
  const raf = useRef(null);

  useEffect(() => {
    const onMove = (e) => { pos.current = { x: e.clientX, y: e.clientY }; };
    window.addEventListener("mousemove", onMove);

    const animate = () => {
      ring.current.x += (pos.current.x - ring.current.x) * 0.12;
      ring.current.y += (pos.current.y - ring.current.y) * 0.12;
      if (dotRef.current) {
        dotRef.current.style.transform = `translate(${pos.current.x - 4}px, ${pos.current.y - 4}px)`;
      }
      if (ringRef.current) {
        ringRef.current.style.transform = `translate(${ring.current.x - 18}px, ${ring.current.y - 18}px)`;
      }
      raf.current = requestAnimationFrame(animate);
    };
    raf.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("mousemove", onMove);
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, []);

  return (
    <>
      {/* Dot */}
      <div ref={dotRef} style={{ position: "fixed", top: 0, left: 0, width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent)", pointerEvents: "none", zIndex: 99999, mixBlendMode: "difference" }} />
      {/* Ring */}
      <div ref={ringRef} style={{ position: "fixed", top: 0, left: 0, width: "36px", height: "36px", borderRadius: "50%", border: "1.5px solid rgba(16,232,184,0.5)", pointerEvents: "none", zIndex: 99998 }} />
    </>
  );
}

const CSS = (t) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;padding:0;font-family:'Plus Jakarta Sans',system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow-x:hidden;cursor:none;background:${t === "dark" ? "#05080f" : "#edf2f7"}}
  #root{min-height:100vh}
  :root{
    --accent:${t === "dark" ? "#10e8b8" : "#0abf97"};
    --accent-dim:${t === "dark" ? "rgba(16,232,184,0.1)" : "rgba(10,191,151,0.1)"};
    --accent-border:${t === "dark" ? "rgba(16,232,184,0.22)" : "rgba(10,191,151,0.3)"};
    --accent-btn-text:${t === "dark" ? "#040c1a" : "#ffffff"};
    --surface:${t === "dark" ? "rgba(255,255,255,0.035)" : "rgba(255,255,255,0.72)"};
    --surface-2:${t === "dark" ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.92)"};
    --nav-bg:${t === "dark" ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.78)"};
    --border:${t === "dark" ? "rgba(255,255,255,0.08)" : "rgba(8,14,36,0.09)"};
    --border-sub:${t === "dark" ? "rgba(255,255,255,0.05)" : "rgba(8,14,36,0.05)"};
    --text:${t === "dark" ? "#e8f4ff" : "#080f28"};
    --text-muted:${t === "dark" ? "rgba(232,244,255,0.52)" : "rgba(8,15,40,0.55)"};
    --text-faint:${t === "dark" ? "rgba(232,244,255,0.22)" : "rgba(8,15,40,0.25)"};
    --shadow:${t === "dark" ? "rgba(0,0,0,0.3)" : "rgba(8,14,36,0.1)"};
    --purple:#6366f1;
    --skel-base:${t === "dark" ? "rgba(255,255,255,0.07)" : "rgba(8,14,36,0.07)"};
    --skel-shine:${t === "dark" ? "rgba(255,255,255,0.13)" : "rgba(8,14,36,0.13)"};
  }
  @keyframes pageEnter{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  @keyframes statusPulse{0%,100%{box-shadow:0 0 0 0 rgba(16,232,184,0.6)}50%{box-shadow:0 0 0 5px rgba(16,232,184,0)}}
  @keyframes statusPulseRed{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.6)}50%{box-shadow:0 0 0 5px rgba(239,68,68,0)}}
  @keyframes checkBlink{0%,100%{opacity:1}50%{opacity:0.25}}
  @keyframes scrollBtnIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  @keyframes skeletonPulse{0%,100%{opacity:0.4}50%{opacity:0.8}}
  @keyframes wiggle{0%,100%{transform:rotate(0deg)}25%{transform:rotate(-8deg)}75%{transform:rotate(8deg)}}
  @keyframes rollIn{from{transform:translateY(100%);opacity:0}to{transform:translateY(0);opacity:1}}

  .page-enter{animation:pageEnter 0.45s cubic-bezier(.22,.68,0,1.1) both}

  /* ── Nav links ── */
  .nav-link{padding:9px 16px;border-radius:11px;text-decoration:none;font-size:0.88rem;font-weight:600;color:var(--text-muted);border:1px solid transparent;transition:all 0.22s ease;font-family:'Plus Jakarta Sans',system-ui,sans-serif;white-space:nowrap}
  .nav-link:hover{color:var(--text);background:var(--surface-2)}
  .nav-link.active{color:var(--text);background:var(--surface-2);border-color:var(--border)}
  .theme-btn{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:var(--surface-2);border:1px solid var(--border);color:var(--text);cursor:pointer;transition:all 0.25s ease}
  .theme-btn:hover{border-color:var(--accent-border);color:var(--accent);transform:rotate(15deg)}
  .scroll-top-btn{position:fixed;bottom:28px;right:28px;z-index:500;width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--accent),#0abf97);border:none;cursor:pointer;box-shadow:0 8px 24px var(--accent-dim);color:var(--accent-btn-text);animation:scrollBtnIn 0.3s ease both;transition:transform 0.22s ease,box-shadow 0.22s ease}
  .scroll-top-btn:hover{transform:translateY(-3px);box-shadow:0 14px 32px rgba(16,232,184,0.35)}
  .scroll-top-btn:active{transform:scale(0.95)}

  /* ── Global Glassmorphism cards ── */
  .glass-card{
    background: ${t === "dark" ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.65)"} !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid ${t === "dark" ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.9)"} !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12), inset 0 1px 0 ${t === "dark" ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.9)"} !important;
  }
  .glass-card:hover{
    background: ${t === "dark" ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.8)"} !important;
    border-color: rgba(16,232,184,0.25) !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.18), 0 0 0 1px rgba(16,232,184,0.12), inset 0 1px 0 ${t === "dark" ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,1)"} !important;
    transform: translateY(-3px);
  }

  /* ── Micro animations ── */
  .micro-btn{transition:transform 0.15s cubic-bezier(.22,.68,0,1.4),box-shadow 0.2s ease,filter 0.2s ease !important}
  .micro-btn:hover{transform:translateY(-2px) !important;filter:brightness(1.08)}
  .micro-btn:active{transform:scale(0.94) !important;filter:brightness(0.96)}

  .icon-wiggle:hover svg{animation:wiggle 0.4s ease}

  /* ── Skeleton loading ── */
  .skeleton{
    background: linear-gradient(90deg, ${t === "dark" ? "rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%" : "rgba(0,0,0,0.06) 0%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.06) 100%"});
    background-size: 200% 100%;
    animation: skeletonSlide 1.5s ease infinite, skeletonPulse 1.5s ease infinite;
    border-radius: 8px;
  }
  @keyframes skeletonSlide{0%{background-position:200% 0}100%{background-position:-200% 0}}
  .skeleton-text{height:14px;border-radius:6px;margin-bottom:8px}
  .skeleton-title{height:24px;border-radius:8px;margin-bottom:12px}
  .skeleton-card{border-radius:22px;padding:28px;border:1px solid var(--border)}
`;

function useBackendStatus() {
  const [s, setS] = useState("checking");
  useEffect(() => {
    const check = async () => {
      try { setS((await checkBackendStatus()) ? "online" : "offline"); } catch { setS("offline"); }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);
  return s;
}

function ScrollToTop() {
  const [v, setV] = useState(false);
  useEffect(() => {
    const fn = () => setV(window.scrollY > 320);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);
  if (!v) return null;
  return (
    <button className="scroll-top-btn" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
    </button>
  );
}

function Navbar({ theme, toggleTheme }) {
  const location = useLocation();
  const isDetect = location.pathname === "/detect";
  const bs = useBackendStatus();
  const cfg = {
    checking: { color: "#f59e0b", label: "Connecting...", anim: "checkBlink 1.2s ease infinite" },
    online: { color: "#10e8b8", label: "Backend Online", anim: "statusPulse 2s ease infinite" },
    offline: { color: "#ef4444", label: "Backend Offline", anim: "statusPulseRed 2s ease infinite" },
  }[bs];

  return (
    <header style={n.navbar}>
      {/* ── LOGO — icon replaces old SV box ── */}
      <div style={n.left}>
        <LogoIcon size={42} />
        <div>
          <div style={{ ...n.logoText, color: "var(--text)" }}>SignVision</div>
          <div style={{ fontSize: "0.72rem", marginTop: "2px", color: "var(--text-faint)" }}>Sign Language Recognition</div>
        </div>
      </div>

      <nav style={n.nav}>
        <NavLink to="/" end className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>Home</NavLink>
        <NavLink to="/detect" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>Detection</NavLink>
        <NavLink to="/about" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>Team</NavLink>
        <NavLink to="/abstract" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>Abstract</NavLink>
        <div style={n.divider} />
        <button className="theme-btn" onClick={toggleTheme} title={theme === "dark" ? "Light Mode" : "Dark Mode"}>
          {theme === "dark"
            ? <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg>
            : <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
          }
        </button>
        <div style={{ ...n.pill, borderColor: `${cfg.color}35`, color: cfg.color }}>
          <span style={{ ...n.dot, background: cfg.color, animation: cfg.anim }} />
          {cfg.label}
        </div>
        {isDetect && (
          <div style={n.live}>
            <span style={{ ...n.dot, background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "statusPulse 1.4s ease infinite" }} />
            LIVE
          </div>
        )}
      </nav>
    </header>
  );
}

function AnimatedRoutes({ theme }) {
  const loc = useLocation();
  const [showing, setShowing] = useState(false);

  useEffect(() => {
    setShowing(false);
    const t = setTimeout(() => setShowing(true), 320);
    return () => clearTimeout(t);
  }, [loc.pathname]);

  const skeletonType = {
    "/": "home", "/detect": "detect", "/about": "about", "/abstract": "abstract"
  }[loc.pathname] || "home";

  return (
    <div key={loc.pathname} className="page-enter">
      {!showing
        ? <SkeletonLoader type={skeletonType} theme={theme} />
        : <Routes location={loc}>
          <Route path="/" element={<Home />} />
          <Route path="/detect" element={<Detect />} />
          <Route path="/about" element={<About />} />
          <Route path="/abstract" element={<Abstract />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      }
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [loaded, setLoaded] = useState(() => !!sessionStorage.getItem("sv_loaded"));

  const handleLoadDone = () => { sessionStorage.setItem("sv_loaded", "true"); setLoaded(true); };

  const bg = theme === "dark"
    ? "radial-gradient(ellipse at 20% 0%,rgba(16,232,184,0.04) 0%,transparent 50%),radial-gradient(ellipse at 80% 100%,rgba(99,102,241,0.05) 0%,transparent 50%),#05080f"
    : "radial-gradient(ellipse at 20% 0%,rgba(10,191,151,0.06) 0%,transparent 50%),radial-gradient(ellipse at 80% 100%,rgba(99,102,241,0.04) 0%,transparent 50%),#edf2f7";

  return (
    <ToastProvider>
      <style>{CSS(theme)}</style>
      {!loaded && <LoadingScreen onDone={handleLoadDone} />}
      <ScrollProgressBar />
      <CustomCursor />
      {/* Vignette — smoothly fades edges into background */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none",
        background: theme === "dark"
          ? "radial-gradient(ellipse at center, transparent 55%, rgba(5,8,15,0.75) 100%)"
          : "radial-gradient(ellipse at center, transparent 55%, rgba(237,242,247,0.75) 100%)",
        transition: "background 0.5s ease",
      }} />
      <BrowserRouter>
        <div style={{ minHeight: "100vh", background: bg, color: "var(--text)", transition: "background 0.5s ease,color 0.3s ease" }}>
          <Navbar theme={theme} toggleTheme={() => setTheme(t => t === "dark" ? "light" : "dark")} />
          <AnimatedRoutes theme={theme} />
          <ScrollToTop />
        </div>
      </BrowserRouter>
    </ToastProvider>
  );
}

const n = {
  navbar: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "16px", flexWrap: "wrap", padding: "12px 24px", margin: "18px 24px", borderRadius: "20px", background: "var(--nav-bg)", border: "1px solid var(--border)", backdropFilter: "blur(20px)", boxShadow: "0 8px 32px var(--shadow)", position: "sticky", top: "18px", zIndex: 100, transition: "all 0.4s ease" },
  left: { display: "flex", alignItems: "center", gap: "12px" },
  logoText: { fontSize: "1.05rem", fontWeight: 800, lineHeight: 1.1, letterSpacing: "-0.01em" },
  nav: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" },
  divider: { width: "1px", height: "20px", background: "var(--border)", margin: "0 4px" },
  pill: { display: "inline-flex", alignItems: "center", gap: "7px", padding: "7px 13px", borderRadius: "999px", background: "var(--surface-2)", border: "1px solid transparent", fontSize: "0.78rem", fontWeight: 600, fontFamily: "'Space Mono',monospace", letterSpacing: "0.03em", whiteSpace: "nowrap", transition: "all 0.4s ease" },
  dot: { width: "7px", height: "7px", borderRadius: "50%", flexShrink: 0 },
  live: { display: "inline-flex", alignItems: "center", gap: "6px", padding: "7px 13px", borderRadius: "999px", background: "var(--accent-dim)", border: "1px solid var(--accent-border)", fontSize: "0.75rem", fontWeight: 700, color: "var(--accent)", letterSpacing: "0.08em", fontFamily: "'Space Mono',monospace" },
};