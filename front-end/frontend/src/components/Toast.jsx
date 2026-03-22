import { useState, useEffect, useCallback, createContext, useContext, useRef } from "react";

const CSS = `
  @keyframes toastIn {
    from { opacity: 0; transform: translateX(110%); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes toastOut {
    from { opacity: 1; transform: translateX(0); }
    to   { opacity: 0; transform: translateX(110%); }
  }
  @keyframes toastProgress {
    from { width: 100%; }
    to   { width: 0%; }
  }
  .toast-enter { animation: toastIn  0.35s cubic-bezier(.22,.68,0,1.2) both; }
  .toast-exit  { animation: toastOut 0.3s ease forwards; }
`;

// ── Context ────────────────────────────────────────────────
export const ToastContext = createContext(null);

export function useToast() {
    return useContext(ToastContext);
}

// ── Toast types config ─────────────────────────────────────
const TYPES = {
    success: {
        color: "#10e8b8", bg: "rgba(16,232,184,0.1)", border: "rgba(16,232,184,0.25)", icon: (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" />
            </svg>
        )
    },
    error: {
        color: "#ef4444", bg: "rgba(239,68,68,0.1)", border: "rgba(239,68,68,0.25)", icon: (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
        )
    },
    info: {
        color: "#6366f1", bg: "rgba(99,102,241,0.1)", border: "rgba(99,102,241,0.25)", icon: (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
        )
    },
    warning: {
        color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.25)", icon: (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
        )
    },
};

// ── Single Toast Item ──────────────────────────────────────
function ToastItem({ toast, onRemove }) {
    const [exiting, setExiting] = useState(false);
    const t = TYPES[toast.type] || TYPES.info;

    const dismiss = useCallback(() => {
        setExiting(true);
        setTimeout(() => onRemove(toast.id), 300);
    }, [toast.id, onRemove]);

    useEffect(() => {
        const timer = setTimeout(dismiss, toast.duration || 3000);
        return () => clearTimeout(timer);
    }, [dismiss, toast.duration]);

    return (
        <div
            className={exiting ? "toast-exit" : "toast-enter"}
            style={{
                display: "flex", alignItems: "flex-start", gap: "10px",
                padding: "13px 14px", borderRadius: "14px",
                background: "rgba(5,8,15,0.92)",
                border: `1px solid ${t.border}`,
                backdropFilter: "blur(16px)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                minWidth: "260px", maxWidth: "340px",
                position: "relative", overflow: "hidden",
                cursor: "pointer",
                fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
            }}
            onClick={dismiss}
        >
            {/* Icon */}
            <div style={{
                width: "28px", height: "28px", borderRadius: "8px", flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: t.bg, color: t.color,
            }}>
                {t.icon}
            </div>

            {/* Text */}
            <div style={{ flex: 1, paddingTop: "2px" }}>
                {toast.title && (
                    <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "#e8f4ff", marginBottom: "2px" }}>
                        {toast.title}
                    </div>
                )}
                <div style={{ fontSize: "0.82rem", color: "rgba(232,244,255,0.65)", lineHeight: 1.4 }}>
                    {toast.message}
                </div>
            </div>

            {/* Progress bar */}
            <div style={{
                position: "absolute", bottom: 0, left: 0, height: "2px",
                background: t.color, opacity: 0.5,
                animation: `toastProgress ${toast.duration || 3000}ms linear forwards`,
                borderRadius: "0 0 14px 14px",
            }} />
        </div>
    );
}

// ── Toast Container ────────────────────────────────────────
export function ToastProvider({ children }) {
    const [toasts, setToasts] = useState([]);
    const idRef = useRef(0);

    const toast = useCallback((message, type = "info", options = {}) => {
        const id = ++idRef.current;
        setToasts(prev => [...prev, { id, message, type, ...options }]);
    }, []);

    const remove = useCallback((id) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    // Convenience methods
    toast.success = (msg, opts) => toast(msg, "success", opts);
    toast.error = (msg, opts) => toast(msg, "error", opts);
    toast.info = (msg, opts) => toast(msg, "info", opts);
    toast.warning = (msg, opts) => toast(msg, "warning", opts);

    return (
        <ToastContext.Provider value={toast}>
            <style>{CSS}</style>
            {children}

            {/* Toast stack — bottom right */}
            <div style={{
                position: "fixed", bottom: "84px", right: "28px",
                zIndex: 9000, display: "flex",
                flexDirection: "column", gap: "10px",
                alignItems: "flex-end", pointerEvents: "none",
            }}>
                {toasts.map(t => (
                    <div key={t.id} style={{ pointerEvents: "auto" }}>
                        <ToastItem toast={t} onRemove={remove} />
                    </div>
                ))}
            </div>
        </ToastContext.Provider>
    );
}