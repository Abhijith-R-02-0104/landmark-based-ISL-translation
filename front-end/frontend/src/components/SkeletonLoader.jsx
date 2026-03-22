const CSS = `
  @keyframes skeletonShimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
  @keyframes skeletonFadeIn { from{opacity:0} to{opacity:1} }
  .sk-wrap { animation: skeletonFadeIn 0.25s ease both; }
  .sk {
    background: linear-gradient(90deg, var(--skel-base) 0%, var(--skel-shine) 50%, var(--skel-base) 100%);
    background-size: 200% 100%;
    animation: skeletonShimmer 1.6s ease infinite;
    border-radius: 10px;
    flex-shrink: 0;
  }
`;

function B({ w = "100%", h = "16px", r = "10px", mb = "0px", delay = "0s" }) {
    return <div className="sk" style={{ width: w, height: h, borderRadius: r, marginBottom: mb, animationDelay: delay }} />;
}

function Card({ children, style = {} }) {
    return (
        <div style={{ borderRadius: "20px", padding: "24px", border: "1px solid var(--border)", background: "var(--surface)", backdropFilter: "blur(12px)", ...style }}>
            {children}
        </div>
    );
}

function HomeSkeleton() {
    return (
        <div style={{ maxWidth: "960px", margin: "0 auto", padding: "60px 24px 48px", display: "flex", flexDirection: "column", alignItems: "center", gap: "20px" }}>
            <B w="260px" h="36px" r="999px" />
            <B w="72%" h="68px" r="14px" delay="0.08s" />
            <B w="54%" h="68px" r="14px" delay="0.1s" />
            <B w="62%" h="22px" r="8px" delay="0.14s" />
            <B w="44%" h="22px" r="8px" delay="0.16s" />
            <div style={{ display: "flex", gap: "12px", marginTop: "8px" }}>
                {["160px", "140px", "140px"].map((w, i) => <B key={i} w={w} h="48px" r="14px" delay={`${0.05 * i}s`} />)}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "16px", width: "100%", maxWidth: "720px", marginTop: "12px" }}>
                {[0, 1, 2].map(i => <B key={i} h="100px" r="20px" delay={`${0.08 * i}s`} />)}
            </div>
        </div>
    );
}

function DetectSkeleton() {
    return (
        <div style={{ maxWidth: "1300px", margin: "0 auto", padding: "0 24px 60px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", paddingBottom: "28px", gap: "16px" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                    <B w="100px" h="14px" />
                    <B w="240px" h="40px" delay="0.08s" />
                </div>
                <B w="160px" h="42px" r="999px" delay="0.12s" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1.45fr 1fr", gap: "20px" }}>
                <B h="540px" r="22px" delay="0.1s" />
                <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                    {[110, 90, 90, 120, 56, 100].map((h, i) => <B key={i} h={`${h}px`} r="16px" delay={`${i * 0.06}s`} />)}
                </div>
            </div>
        </div>
    );
}

function AboutSkeleton() {
    return (
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "48px 24px 80px" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "18px", marginBottom: "52px" }}>
                <B w="110px" h="110px" r="50%" />
                <B w="220px" h="28px" r="999px" delay="0.08s" />
                <B w="52%" h="62px" delay="0.12s" />
                <Card style={{ width: "100%", maxWidth: "640px" }}>
                    <B w="80px" h="12px" mb="10px" delay="0.16s" />
                    <B h="18px" mb="8px" delay="0.18s" />
                    <B w="70%" h="18px" delay="0.2s" />
                </Card>
                <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", justifyContent: "center" }}>
                    {[0, 1, 2, 3].map(i => <B key={i} w="90px" h="64px" r="16px" delay={`${i * 0.07}s`} />)}
                </div>
            </div>
            <div style={{ maxWidth: "900px", margin: "0 auto" }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "16px", marginBottom: "36px" }}>
                    {[0, 1, 2, 3].map(i => <B key={i} h="170px" r="22px" delay={`${i * 0.07}s`} />)}
                </div>
                <B h="120px" r="22px" mb="14px" delay="0.2s" />
                <B h="120px" r="22px" delay="0.25s" />
            </div>
        </div>
    );
}

function AbstractSkeleton() {
    return (
        <div style={{ maxWidth: "900px", margin: "0 auto", padding: "0 24px 80px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "20px 0 32px" }}>
                <B w="130px" h="38px" r="10px" />
                <B w="180px" h="34px" r="999px" delay="0.06s" />
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "18px", marginBottom: "40px" }}>
                <B w="220px" h="32px" r="999px" />
                <B w="58%" h="64px" delay="0.08s" />
                <Card style={{ width: "100%", maxWidth: "720px" }}>
                    <B w="80px" h="12px" mb="12px" delay="0.12s" />
                    <B h="18px" mb="8px" delay="0.14s" />
                    <B w="60%" h="18px" mb="12px" delay="0.16s" />
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                        {[0, 1, 2, 3].map(i => <B key={i} w="140px" h="28px" r="999px" delay={`${i * 0.05}s`} />)}
                    </div>
                </Card>
            </div>
            {[220, 200, 240, 200, 220, 160].map((h, i) => (
                <div key={i} style={{ marginBottom: "20px" }}><B h={`${h}px`} r="22px" delay={`${i * 0.08}s`} /></div>
            ))}
        </div>
    );
}

const MAP = { home: HomeSkeleton, detect: DetectSkeleton, about: AboutSkeleton, abstract: AbstractSkeleton };

export default function SkeletonLoader({ type = "home" }) {
    const Comp = MAP[type] || HomeSkeleton;
    return (
        <>
            <style>{CSS}</style>
            <div className="sk-wrap"><Comp /></div>
        </>
    );
}