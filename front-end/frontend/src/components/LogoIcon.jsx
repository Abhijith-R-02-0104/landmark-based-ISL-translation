// SignVision Logo Icon — reusable at any size
// Usage: <LogoIcon size={42} /> or <LogoIcon size={80} />

export default function LogoIcon({ size = 42 }) {
    return (
        <svg
            width={size}
            height={Math.round(size * (90 / 80))}
            viewBox="0 0 80 90"
            xmlns="http://www.w3.org/2000/svg"
            style={{ flexShrink: 0, display: "block" }}
        >
            <defs>
                <linearGradient id="sv_block" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#13edd4" />
                    <stop offset="100%" stopColor="#0784a8" />
                </linearGradient>
                <linearGradient id="sv_shine" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="white" stopOpacity="0.14" />
                    <stop offset="100%" stopColor="white" stopOpacity="0" />
                </linearGradient>
            </defs>

            {/* Outer glow rings */}
            <rect x="0" y="0" width="80" height="90" rx="22" fill="none" stroke="#10e8b8" strokeWidth="0.6" strokeOpacity="0.22" />
            <rect x="1.5" y="1.5" width="77" height="87" rx="21" fill="none" stroke="#10e8b8" strokeWidth="0.4" strokeOpacity="0.1" />

            {/* Main teal block */}
            <rect x="3" y="3" width="74" height="84" rx="20" fill="url(#sv_block)" />

            {/* Glass shine on top */}
            <rect x="7" y="7" width="66" height="36" rx="16" fill="url(#sv_shine)" />

            {/* Bottom subtle shadow */}
            <rect x="3" y="56" width="74" height="31" rx="20" fill="#020810" fillOpacity="0.1" />

            {/* Inner border */}
            <rect x="3" y="3" width="74" height="84" rx="20" fill="none" stroke="white" strokeWidth="0.5" strokeOpacity="0.12" />

            {/* Hand skeleton — full black lines */}
            <g stroke="#020408" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.75">
                {/* Wrist to palm bases */}
                <line x1="40" y1="76" x2="22" y2="53" />
                <line x1="40" y1="76" x2="31" y2="51" />
                <line x1="40" y1="76" x2="43" y2="51" />
                <line x1="40" y1="76" x2="55" y2="55" />
                {/* Palm bar */}
                <line x1="22" y1="53" x2="31" y2="51" />
                <line x1="31" y1="51" x2="43" y2="51" />
                <line x1="43" y1="51" x2="55" y2="55" />
                {/* Index finger */}
                <line x1="22" y1="53" x2="19" y2="38" />
                <line x1="19" y1="38" x2="17" y2="23" />
                {/* Middle finger */}
                <line x1="31" y1="51" x2="30" y2="35" />
                <line x1="30" y1="35" x2="29" y2="19" />
                {/* Ring finger */}
                <line x1="43" y1="51" x2="43" y2="34" />
                <line x1="43" y1="34" x2="43" y2="17" />
                {/* Pinky */}
                <line x1="55" y1="55" x2="56" y2="40" />
                <line x1="56" y1="40" x2="57" y2="27" />
                {/* Thumb */}
                <line x1="22" y1="53" x2="13" y2="43" />
                <line x1="13" y1="43" x2="10" y2="31" />
            </g>

            {/* All landmark dots — full black */}
            <g fill="#020408">
                {/* Wrist */}
                <circle cx="40" cy="76" r="3.4" />
                {/* Palm bases */}
                <circle cx="22" cy="53" r="2.6" />
                <circle cx="31" cy="51" r="2.6" />
                <circle cx="43" cy="51" r="2.6" />
                <circle cx="55" cy="55" r="2.6" />
                {/* Mid knuckles */}
                <circle cx="19" cy="38" r="2.2" />
                <circle cx="30" cy="35" r="2.2" />
                <circle cx="43" cy="34" r="2.2" />
                <circle cx="56" cy="40" r="2.2" />
                {/* Thumb mid */}
                <circle cx="13" cy="43" r="1.9" />
                {/* Fingertips */}
                <circle cx="17" cy="23" r="3.2" />
                <circle cx="29" cy="19" r="3.2" />
                <circle cx="43" cy="17" r="3.6" />
                <circle cx="57" cy="27" r="3.2" />
                {/* Thumb tip */}
                <circle cx="10" cy="31" r="2.8" />
            </g>
        </svg>
    );
}