import { useEffect, useRef } from "react";

export default function Particles() {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        let animId;
        let W = window.innerWidth;
        let H = window.innerHeight;
        canvas.width = W;
        canvas.height = H;

        // Generate particles — landmark-style dots
        const COUNT = 55;
        const particles = Array.from({ length: COUNT }, () => ({
            x: Math.random() * W,
            y: Math.random() * H,
            r: Math.random() * 2.2 + 0.8,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            alpha: Math.random() * 0.45 + 0.1,
            pulse: Math.random() * Math.PI * 2,
            pulseSpeed: Math.random() * 0.018 + 0.006,
            connected: [],
        }));

        const TEAL = [16, 232, 184];
        const PURPLE = [99, 102, 241];

        const resize = () => {
            W = window.innerWidth;
            H = window.innerHeight;
            canvas.width = W;
            canvas.height = H;
        };
        window.addEventListener("resize", resize);

        const draw = () => {
            ctx.clearRect(0, 0, W, H);

            // Draw connection lines between nearby particles
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        const opacity = (1 - dist / 120) * 0.12;
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(${TEAL[0]},${TEAL[1]},${TEAL[2]},${opacity})`;
                        ctx.lineWidth = 0.6;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Draw dots
            particles.forEach((p, i) => {
                p.pulse += p.pulseSpeed;
                const pulsedAlpha = p.alpha * (0.7 + 0.3 * Math.sin(p.pulse));
                const pulsedR = p.r * (0.9 + 0.1 * Math.sin(p.pulse));

                // Alternate between teal and purple for variety
                const color = i % 7 === 0 ? PURPLE : TEAL;

                // Outer glow
                const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pulsedR * 3.5);
                grd.addColorStop(0, `rgba(${color[0]},${color[1]},${color[2]},${pulsedAlpha * 0.35})`);
                grd.addColorStop(1, `rgba(${color[0]},${color[1]},${color[2]},0)`);
                ctx.beginPath();
                ctx.arc(p.x, p.y, pulsedR * 3.5, 0, Math.PI * 2);
                ctx.fillStyle = grd;
                ctx.fill();

                // Core dot
                ctx.beginPath();
                ctx.arc(p.x, p.y, pulsedR, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${pulsedAlpha})`;
                ctx.fill();

                // Move
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < -20) p.x = W + 20;
                if (p.x > W + 20) p.x = -20;
                if (p.y < -20) p.y = H + 20;
                if (p.y > H + 20) p.y = -20;
            });

            animId = requestAnimationFrame(draw);
        };

        draw();
        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener("resize", resize);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: "fixed",
                inset: 0,
                zIndex: 0,
                pointerEvents: "none",
                opacity: 0.6,
            }}
        />
    );
}