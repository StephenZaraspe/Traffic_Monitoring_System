/**
 * MMDA TRAFFIC INTELLIGENCE — Dashboard (Thesis Edition)
 * Supports two modes:
 *   1. Video Upload  — upload .mp4, GPU processes it, plays finished annotated video
 *   2. Live Webcam   — backend opens camera via OpenCV, streams annotated MJPEG live
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";

const VEHICLE_TYPES = ["cars", "motorcycle", "trucks", "trike", "bus", "jeepney", "e-jeepney"];
const CLASS_COLORS = {
    cars: "#00d4ff", motorcycle: "#ff6b35", trucks: "#ffd700",
    trike: "#a78bfa", bus: "#34d399", jeepney: "#fb7185", "e-jeepney": "#67e8f9",
};
const MODELS = [
    { id: "baseline", label: "Baseline", desc: "699-frame raw set" },
    { id: "distilled", label: "Distilled", desc: "100 synthetic canvases - OD3" },
    { id: "finetuned", label: "Distilled + Fine-tuned", desc: "OD3 + 10% Domain Bridge" },
];
const ABLATION = [
    { label: "Cars", ap: 98.6, color: "#00d4ff" },
    { label: "Motorcycle", ap: 97.0, color: "#ff6b35" },
    { label: "E-Jeepney", ap: 100, color: "#67e8f9" },
    { label: "Jeepney", ap: 91.2, color: "#fb7185" },
    { label: "Trike", ap: 84.3, color: "#a78bfa" },
    { label: "Bus", ap: 93.5, color: "#34d399" },
    { label: "Trucks", ap: 95.1, color: "#ffd700" },
];

function useCountUp(target) {
    const [val, setVal] = useState(0);
    const prev = useRef(0);
    useEffect(() => {
        const start = prev.current, diff = target - start;
        if (diff === 0) return;
        const t0 = Date.now();
        const tick = () => {
            const p = Math.min(1, (Date.now() - t0) / 800);
            setVal(Math.round(start + (1 - Math.pow(1 - p, 3)) * diff));
            if (p < 1) requestAnimationFrame(tick);
            else { prev.current = target; setVal(target); }
        };
        requestAnimationFrame(tick);
    }, [target]);
    return val;
}

function StatCard({ label, value, accent, sub }) {
    const animated = useCountUp(value);
    return (
        <div style={{
            flex: 1, minWidth: 140,
            background: "linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))",
            border: "1px solid " + accent + "25", borderRadius: 14, padding: "20px 22px",
            position: "relative", overflow: "hidden"
        }}>
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0, height: 2,
                background: "linear-gradient(90deg,transparent," + accent + "80,transparent)"
            }} />
            <div style={{
                position: "absolute", inset: 0,
                background: "radial-gradient(ellipse 80% 60% at 0% 0%," + accent + "08,transparent)",
                pointerEvents: "none"
            }} />
            <div style={{
                fontFamily: "'Space Mono',monospace", fontSize: 38, fontWeight: 700,
                color: accent, lineHeight: 1, letterSpacing: -1
            }}>{animated.toLocaleString()}</div>
            <div style={{
                fontSize: 10, color: "#64748b", marginTop: 8,
                letterSpacing: 2.5, textTransform: "uppercase"
            }}>{label}</div>
            {sub && <div style={{ fontSize: 9, color: "#334155", marginTop: 3 }}>{sub}</div>}
        </div>
    );
}

function AblationPanel() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {ABLATION.map((a, i) => (
                <div key={a.label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{
                        fontSize: 9, color: "#475569", fontFamily: "Space Mono",
                        width: 72, flexShrink: 0
                    }}>{a.label}</div>
                    <div style={{
                        flex: 1, height: 5, background: "rgba(255,255,255,0.04)",
                        borderRadius: 3, overflow: "hidden"
                    }}>
                        <div style={{
                            width: a.ap + "%", height: "100%", background: a.color,
                            borderRadius: 3, opacity: 0.8,
                            animation: "growBar 1s ease " + (i * 0.08) + "s both"
                        }} />
                    </div>
                    <div style={{
                        fontSize: 9, color: a.ap >= 95 ? "#34d399" : "#94a3b8",
                        fontFamily: "Space Mono", width: 36, textAlign: "right"
                    }}>
                        {a.ap === 100 ? "100%" : a.ap + "%"}
                    </div>
                </div>
            ))}
            <div style={{
                marginTop: 10, paddingTop: 10,
                borderTop: "1px solid rgba(255,255,255,0.04)",
                display: "flex", justifyContent: "space-between"
            }}>
                <span style={{ fontSize: 8, color: "#334155", fontFamily: "Space Mono" }}>
                    Overall CER (human-validated)
                </span>
                <span style={{
                    fontSize: 10, color: "#34d399",
                    fontFamily: "Space Mono", fontWeight: 700
                }}>9.69%</span>
            </div>
        </div>
    );
}

// ---- UPLOAD MODAL -----------------------------------------------------------
function UploadModal({ onConnect }) {
    const [localUrl, setLocalUrl] = useState("http://localhost:8080");
    const [cfUrl, setCfUrl] = useState("");
    const [file, setFile] = useState(null);
    const [uploadState, setUploadState] = useState("idle");
    const [uploadPct, setUploadPct] = useState(0);
    const [msg, setMsg] = useState("");
    const [dragOver, setDragOver] = useState(false);
    const fileRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;
        const ext = f.name.split(".").pop().toLowerCase();
        if (!["mp4", "avi", "mov", "mkv", "webm"].includes(ext)) {
            setMsg("Unsupported format. Use MP4, AVI, MOV, MKV or WEBM.");
            setUploadState("error"); return;
        }
        setFile(f); setUploadState("idle"); setMsg("");
    };

    const handleUpload = () => {
        if (!file) { setMsg("Select a video file first."); setUploadState("error"); return; }
        const target = (localUrl.trim() || "http://localhost:8080").replace(/\/$/, "");
        setUploadState("uploading"); setUploadPct(0);
        setMsg("Uploading to " + target + "...");
        const fd = new FormData();
        fd.append("file", file);
        const xhr = new XMLHttpRequest();
        xhr.open("POST", target + "/api/upload/video");
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) setUploadPct(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
            if (xhr.status === 200) {
                let res;
                try { res = JSON.parse(xhr.responseText); } catch (e) {
                    setUploadState("error"); setMsg("Bad server response"); return;
                }
                if (res.error) { setUploadState("error"); setMsg(res.error); return; }
                setUploadState("done");
                setMsg("Uploaded " + res.filename + " (" + res.size_mb + " MB). Processing on GPU...");
                const connectUrl = cfUrl.trim() || target;
                onConnect({ url: connectUrl, localUrl: target, webcam: false });
            } else {
                setUploadState("error"); setMsg("Server error " + xhr.status);
            }
        };
        xhr.onerror = () => {
            setUploadState("error");
            setMsg("Cannot reach " + target + " - is uvicorn running on port 8080?");
        };
        xhr.send(fd);
    };

    const handleWebcam = () => {
        const target = (localUrl.trim() || "http://localhost:8080").replace(/\/$/, "");
        onConnect({ url: cfUrl.trim() || target, localUrl: target, webcam: true });
    };

    const stateColor = uploadState === "uploading" ? "#00d4ff"
        : uploadState === "done" ? "#34d399"
            : uploadState === "error" ? "#fb7185" : "#475569";

    const inp = {
        width: "100%", background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(0,212,255,0.15)", borderRadius: 8,
        color: "#e2e8f0", padding: "11px 13px", fontSize: 12,
        fontFamily: "Space Mono", outline: "none", boxSizing: "border-box",
    };

    return (
        <div style={{
            position: "fixed", inset: 0, background: "rgba(2,6,18,0.98)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 1000, fontFamily: "'Syne',sans-serif"
        }}>
            <div style={{
                background: "#070d1c", border: "1px solid rgba(0,212,255,0.15)",
                borderRadius: 18, padding: 44, width: 520, maxWidth: "94vw",
                boxShadow: "0 0 80px rgba(0,100,255,0.08)", maxHeight: "92vh", overflowY: "auto"
            }}>

                <div style={{
                    fontSize: 9, color: "#00d4ff", letterSpacing: 3,
                    marginBottom: 10, fontFamily: "Space Mono"
                }}>
                    MMDA TRAFFIC INTELLIGENCE - THESIS EDITION
                </div>
                <div style={{ fontSize: 22, fontWeight: 800, color: "#f1f5f9", marginBottom: 4 }}>
                    System Configuration
                </div>
                <div style={{ fontSize: 12, color: "#475569", marginBottom: 26, lineHeight: 1.7 }}>
                    Upload CCTV footage for GPU processing, or use live webcam counting.
                </div>

                <div style={{ marginBottom: 14 }}>
                    <div style={{
                        fontSize: 9, color: "#475569", letterSpacing: 2,
                        marginBottom: 7, fontFamily: "Space Mono"
                    }}>LOCAL BACKEND URL</div>
                    <input value={localUrl} onChange={e => setLocalUrl(e.target.value)}
                        placeholder="http://localhost:8080" style={inp} />
                    <div style={{ fontSize: 8, color: "#1e3a5f", fontFamily: "Space Mono", marginTop: 4 }}>
                        Upload and webcam go here directly - bypasses Cloudflare
                    </div>
                </div>

                <div style={{ marginBottom: 18 }}>
                    <div style={{
                        fontSize: 9, color: "#475569", letterSpacing: 2,
                        marginBottom: 7, fontFamily: "Space Mono"
                    }}>
                        CLOUDFLARE URL (optional)
                    </div>
                    <input value={cfUrl} onChange={e => setCfUrl(e.target.value)}
                        placeholder="https://xxxx.trycloudflare.com  (leave blank for local)"
                        style={inp} />
                </div>

                {/* DROP ZONE */}
                <div style={{ marginBottom: 14 }}>
                    <div style={{
                        fontSize: 9, color: "#475569", letterSpacing: 2,
                        marginBottom: 7, fontFamily: "Space Mono"
                    }}>VIDEO FILE</div>
                    <div
                        onClick={() => fileRef.current && fileRef.current.click()}
                        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
                        style={{
                            border: "1.5px dashed " + (dragOver ? "rgba(0,212,255,0.6)" : "rgba(0,212,255,0.2)"),
                            borderRadius: 10, padding: "24px 16px", textAlign: "center", cursor: "pointer",
                            transition: "all 0.2s",
                            background: dragOver ? "rgba(0,212,255,0.04)" : "rgba(255,255,255,0.01)"
                        }}>
                        <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.3 }}>&#x1F4F9;</div>
                        {file ? (
                            <div>
                                <div style={{
                                    fontSize: 12, color: "#00d4ff",
                                    fontFamily: "Space Mono", fontWeight: 700
                                }}>{file.name}</div>
                                <div style={{
                                    fontSize: 9, color: "#334155",
                                    fontFamily: "Space Mono", marginTop: 4
                                }}>
                                    {(file.size / 1024 / 1024).toFixed(1)} MB
                                </div>
                            </div>
                        ) : (
                            <div>
                                <div style={{ fontSize: 12, color: "#475569", fontFamily: "Space Mono" }}>
                                    Drag and drop or click to select
                                </div>
                                <div style={{
                                    fontSize: 9, color: "#334155",
                                    fontFamily: "Space Mono", marginTop: 4
                                }}>
                                    MP4 - AVI - MOV - MKV - WEBM
                                </div>
                            </div>
                        )}
                    </div>
                    <input ref={fileRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm"
                        style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
                </div>

                {/* Upload progress */}
                {uploadState !== "idle" && (
                    <div style={{ marginBottom: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                            <span style={{ fontSize: 9, color: stateColor, fontFamily: "Space Mono" }}>
                                {msg}
                            </span>
                            {uploadState === "uploading" && (
                                <span style={{ fontSize: 9, color: "#00d4ff", fontFamily: "Space Mono" }}>
                                    {uploadPct}%
                                </span>
                            )}
                        </div>
                        {uploadState === "uploading" && (
                            <div style={{
                                height: 4, background: "rgba(255,255,255,0.05)",
                                borderRadius: 2, overflow: "hidden"
                            }}>
                                <div style={{
                                    width: uploadPct + "%", height: "100%",
                                    background: "linear-gradient(90deg,#00d4ff,#0066ff)",
                                    borderRadius: 2, transition: "width 0.3s"
                                }} />
                            </div>
                        )}
                    </div>
                )}

                {/* Quick start */}
                <div style={{
                    background: "rgba(0,0,0,0.3)",
                    border: "1px solid rgba(255,255,255,0.05)", borderRadius: 8,
                    padding: "11px 14px", marginBottom: 18,
                    fontSize: 10, color: "#475569", fontFamily: "Space Mono", lineHeight: 1.9
                }}>
                    <div style={{ color: "#f6821f", marginBottom: 4 }}>QUICK START</div>
                    <div>1. uvicorn main_api:app --host 0.0.0.0 --port 8080</div>
                    <div>2. Select video or click Use Webcam below</div>
                </div>

                {/* Upload button */}
                <button
                    onClick={handleUpload}
                    disabled={!file || uploadState === "uploading" || uploadState === "done"}
                    style={{
                        width: "100%",
                        background: file ? "linear-gradient(135deg,rgba(0,212,255,0.12),rgba(0,102,255,0.12))"
                            : "rgba(255,255,255,0.03)",
                        border: "1px solid " + (file ? "rgba(0,212,255,0.35)" : "rgba(255,255,255,0.06)"),
                        borderRadius: 10, color: file ? "#00d4ff" : "#334155",
                        padding: "13px", fontSize: 11, fontWeight: 700,
                        cursor: file ? "pointer" : "not-allowed",
                        fontFamily: "Space Mono", letterSpacing: 2
                    }}>
                    {uploadState === "uploading" ? "UPLOADING " + uploadPct + "%"
                        : uploadState === "done" ? "PROCESSING ON GPU..."
                            : "UPLOAD AND PROCESS"}
                </button>

                {/* Divider */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 18 }}>
                    <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }} />
                    <span style={{ fontSize: 9, color: "#334155", fontFamily: "Space Mono" }}>OR</span>
                    <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }} />
                </div>

                {/* Webcam button */}
                <button
                    onClick={handleWebcam}
                    style={{
                        width: "100%", marginTop: 14,
                        background: "linear-gradient(135deg,rgba(52,211,153,0.08),rgba(0,200,100,0.08))",
                        border: "1px solid rgba(52,211,153,0.25)", borderRadius: 10,
                        color: "#34d399", padding: "13px", fontSize: 11, fontWeight: 700,
                        cursor: "pointer", fontFamily: "Space Mono", letterSpacing: 2
                    }}>
                    &#9679; USE WEBCAM (live counting)
                </button>
                <div style={{
                    fontSize: 8, color: "#1e3a5f", fontFamily: "Space Mono",
                    textAlign: "center", marginTop: 6
                }}>
                    Backend opens your camera via OpenCV - smooth live annotated feed
                </div>

                <div style={{ textAlign: "center", marginTop: 14 }}>
                    <span onClick={() => onConnect({ url: "", localUrl: "http://localhost:8080", webcam: false })}
                        style={{ fontSize: 9, color: "#334155", fontFamily: "Space Mono", cursor: "pointer" }}>
                        or run in <span style={{ color: "#ffd700" }}>DEMO MODE</span>
                    </span>
                </div>
            </div>
        </div>
    );
}

// ---- MAIN DASHBOARD ---------------------------------------------------------
export default function TrafficDashboard() {
    const [showUpload, setShowUpload] = useState(true);
    const [demoMode, setDemoMode] = useState(false);
    const [model, setModel] = useState("finetuned");
    const [webcamMode, setWebcamMode] = useState(false);
    const [webcamOn, setWebcamOn] = useState(false);
    const [procState, setProcState] = useState("idle");
    const [procPct, setProcPct] = useState(0);
    const [procMsg, setProcMsg] = useState("");
    const [counts, setCounts] = useState({});
    const [inCount, setInCount] = useState(0);
    const [outCount, setOutCount] = useState(0);
    const [log, setLog] = useState([]);
    const [uptime, setUptime] = useState(0);
    const [connected, setConnected] = useState(false);

    const wsRef = useRef(null);
    const pollRef = useRef(null);
    const startTs = useRef(Date.now());
    const videoRef = useRef(null);
    const localUrlRef = useRef("http://localhost:8080");

    useEffect(() => {
        const t = setInterval(() =>
            setUptime(Math.floor((Date.now() - startTs.current) / 1000)), 1000);
        return () => clearInterval(t);
    }, []);

    const handleEvent = useCallback((event) => {
        const { vehicleType: v, direction: d } = event;
        setCounts(prev => ({ ...prev, [v]: (prev[v] || 0) + 1 }));
        if (d === "IN") setInCount(p => p + 1);
        if (d === "OUT") setOutCount(p => p + 1);
        setLog(prev => [event, ...prev].slice(0, 20));
    }, []);

    const startPolling = useCallback((localUrl) => {
        localUrlRef.current = localUrl;
        if (pollRef.current) clearInterval(pollRef.current);

        pollRef.current = setInterval(async () => {
            const url = localUrlRef.current;
            try {
                const r = await fetch(url + "/api/status");
                const data = await r.json();
                setProcState(data.state);
                setProcPct(data.progress || 0);
                setProcMsg(data.message || "");

                if (data.state === "done") {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    try {
                        const sr = await fetch(url + "/api/traffic/summary");
                        const sd = await sr.json();
                        setCounts(sd.counts || {});
                        setInCount(sd.inbound || 0);
                        setOutCount(sd.outbound || 0);
                    } catch (e) { }
                    try {
                        const lr = await fetch(url + "/api/traffic/log?limit=20");
                        const ld = await lr.json();
                        setLog(ld);
                    } catch (e) { }
                    setTimeout(() => {
                        if (videoRef.current) {
                            const src = url + "/api/video/output?t=" + Date.now();
                            videoRef.current.src = src;
                            videoRef.current.load();
                            videoRef.current.play().catch(e => console.log("Autoplay blocked:", e));
                        }
                    }, 500);
                }
            } catch (e) { }
        }, 1000);
    }, []);

    const handleConnect = useCallback(async (cfg) => {
        setShowUpload(false);
        if (!cfg.url && !cfg.localUrl && !cfg.webcam) {
            setDemoMode(true); setProcState("done"); return;
        }
        const local = cfg.localUrl || "http://localhost:8080";
        localUrlRef.current = local;

        // WebSocket for detection events
        const wsBase = (cfg.url || local).replace(/^https?/, m => m === "https" ? "wss" : "ws");
        const connectWs = () => {
            const ws = new WebSocket(wsBase + "/ws/traffic");
            ws.onopen = () => setConnected(true);
            ws.onclose = () => { setConnected(false); setTimeout(connectWs, 3000); };
            ws.onmessage = (e) => {
                try { const ev = JSON.parse(e.data); if (ev.type !== "ping") handleEvent(ev); }
                catch (err) { }
            };
            wsRef.current = ws;
        };
        connectWs();

        if (cfg.webcam) {
            setWebcamMode(true);
            try {
                const r = await fetch(local + "/api/webcam/start", { method: "POST" });
                const d = await r.json();
                if (d.error) { alert("Webcam error: " + d.error); return; }
                setWebcamOn(true);
                setProcState("webcam");
                setConnected(true);
            } catch (e) {
                alert("Cannot reach backend: " + e.message);
            }
        } else {
            startPolling(local);
        }
    }, [handleEvent, startPolling]);

    const handleStopWebcam = async () => {
        const url = localUrlRef.current;
        try {
            await fetch(url + "/api/webcam/stop", { method: "POST" });
        } catch (e) { }
        setWebcamOn(false);
        setWebcamMode(false);
        setProcState("idle");
        setShowUpload(true);
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => () => {
        wsRef.current && wsRef.current.close();
        pollRef.current && clearInterval(pollRef.current);
    }, []);

    const total = inCount + outCount;
    const activeClasses = useMemo(() =>
        Object.keys(counts).filter(k => counts[k] > 0).length, [counts]);
    const uptimeFmt = String(Math.floor(uptime / 3600)).padStart(2, "0") + ":" +
        String(Math.floor((uptime % 3600) / 60)).padStart(2, "0") + ":" +
        String(uptime % 60).padStart(2, "0");
    const statusColor = demoMode ? "#ffd700" : connected ? "#34d399" : "#ef4444";
    const activeModel = MODELS.find(m => m.id === model);

    return (
        <div>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        @keyframes pulseGlow{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.6)}}
        @keyframes slideIn{from{opacity:0;transform:translateY(-5px)}to{opacity:1;transform:translateY(0)}}
        @keyframes scanline{from{transform:translateY(-100vh)}to{transform:translateY(100vh)}}
        @keyframes growBar{from{width:0}}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:2px}
        select,option{background:#060c18!important;color:#e2e8f0}
        input::placeholder{color:#1a2f4a}
        button:active{transform:scale(0.97)}
        video{display:block;background:#000}
      `}</style>

            {showUpload && <UploadModal onConnect={handleConnect} />}

            <div style={{
                minHeight: "100vh", background: "#060c18",
                fontFamily: "'Syne',sans-serif", color: "#e2e8f0", paddingBottom: 32
            }}>

                {/* Bg effects */}
                <div style={{
                    position: "fixed", inset: 0, pointerEvents: "none",
                    zIndex: 0, overflow: "hidden"
                }}>
                    <div style={{
                        position: "absolute", width: "100%", height: "3px",
                        background: "linear-gradient(transparent,rgba(0,212,255,0.025),transparent)",
                        animation: "scanline 10s linear infinite"
                    }} />
                </div>
                <div style={{
                    position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
                    backgroundImage: "linear-gradient(rgba(0,212,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.025) 1px,transparent 1px)",
                    backgroundSize: "48px 48px",
                    maskImage: "radial-gradient(ellipse 80% 60% at 50% 0%,black 30%,transparent 80%)"
                }} />

                {/* HEADER */}
                <header style={{
                    borderBottom: "1px solid rgba(0,212,255,0.08)", padding: "13px 28px",
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    background: "rgba(4,8,18,0.9)", backdropFilter: "blur(28px)",
                    position: "sticky", top: 0, zIndex: 20
                }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                        <div style={{ position: "relative" }}>
                            <div style={{
                                width: 40, height: 40,
                                background: "linear-gradient(135deg,rgba(0,212,255,0.15),rgba(0,80,200,0.1))",
                                border: "1px solid rgba(0,212,255,0.25)", borderRadius: 11,
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: 16, fontWeight: 800, color: "#00d4ff"
                            }}>T</div>
                            <div style={{
                                position: "absolute", bottom: -2, right: -2,
                                width: 11, height: 11, borderRadius: "50%", background: statusColor,
                                border: "2px solid #060c18", animation: "pulseGlow 2s infinite"
                            }} />
                        </div>
                        <div>
                            <div style={{ fontSize: 14, fontWeight: 800, letterSpacing: 1.5, color: "#f1f5f9" }}>
                                MMDA TRAFFIC INTELLIGENCE
                            </div>
                            <div style={{
                                fontSize: 9, color: "#334155",
                                letterSpacing: 2.5, fontFamily: "Space Mono"
                            }}>
                                YOLOv8s - OC-SORT - OD3 Distillation - TIP MANILA 2025
                            </div>
                        </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                        <div style={{
                            display: "flex", flexDirection: "column",
                            gap: 3, alignItems: "flex-end"
                        }}>
                            <div style={{
                                fontSize: 8, color: "#334155",
                                fontFamily: "Space Mono", letterSpacing: 2
                            }}>MODEL</div>
                            <select value={model} onChange={e => setModel(e.target.value)} style={{
                                background: "rgba(0,212,255,0.05)",
                                border: "1px solid rgba(0,212,255,0.18)", borderRadius: 6,
                                color: "#00d4ff", padding: "5px 24px 5px 9px", fontSize: 10,
                                fontFamily: "Space Mono", cursor: "pointer", outline: "none",
                                appearance: "none",
                                backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%2300d4ff'/%3E%3C/svg%3E\")",
                                backgroundRepeat: "no-repeat", backgroundPosition: "right 7px center"
                            }}>
                                {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                            </select>
                        </div>

                        {webcamMode && webcamOn ? (
                            <button onClick={handleStopWebcam} style={{
                                background: "rgba(251,113,133,0.1)",
                                border: "1px solid rgba(251,113,133,0.3)",
                                borderRadius: 7, color: "#fb7185", padding: "7px 14px",
                                fontSize: 9, cursor: "pointer", fontFamily: "Space Mono"
                            }}>
                                STOP WEBCAM
                            </button>
                        ) : (
                            <button onClick={() => setShowUpload(true)} style={{
                                background: "rgba(255,255,255,0.04)",
                                border: "1px solid rgba(255,255,255,0.07)",
                                borderRadius: 7, color: "#475569", padding: "7px 14px",
                                fontSize: 9, cursor: "pointer", fontFamily: "Space Mono"
                            }}>
                                UPLOAD NEW
                            </button>
                        )}
                        <div style={{ fontFamily: "Space Mono", fontSize: 10, color: "#1e3a5f" }}>
                            UP {uptimeFmt}
                        </div>
                    </div>
                </header>

                <main style={{ padding: "20px 28px", position: "relative", zIndex: 1 }}>

                    {/* Model banner */}
                    <div style={{
                        display: "flex", alignItems: "center", gap: 14,
                        padding: "10px 18px", background: "rgba(0,212,255,0.025)",
                        border: "1px solid rgba(0,212,255,0.07)", borderRadius: 10, marginBottom: 20
                    }}>
                        <div style={{
                            fontSize: 8, color: "#334155",
                            fontFamily: "Space Mono", letterSpacing: 2
                        }}>MODEL</div>
                        <div style={{
                            fontSize: 11, color: "#00d4ff",
                            fontFamily: "Space Mono", fontWeight: 700
                        }}>
                            {activeModel && activeModel.label}
                        </div>
                        <div style={{
                            width: 1, height: 12,
                            background: "rgba(255,255,255,0.05)"
                        }} />
                        <div style={{ fontSize: 9, color: "#334155", fontFamily: "Space Mono" }}>
                            {activeModel && activeModel.desc}
                        </div>
                        <div style={{
                            marginLeft: "auto", display: "flex",
                            alignItems: "center", gap: 6
                        }}>
                            {webcamMode && webcamOn && (
                                <div style={{
                                    fontSize: 8, color: "#34d399",
                                    fontFamily: "Space Mono", marginRight: 8,
                                    background: "rgba(52,211,153,0.1)",
                                    padding: "3px 8px", borderRadius: 4,
                                    border: "1px solid rgba(52,211,153,0.2)"
                                }}>
                                    LIVE WEBCAM
                                </div>
                            )}
                            <div style={{
                                width: 6, height: 6, borderRadius: "50%",
                                background: statusColor, animation: "pulseGlow 2s infinite"
                            }} />
                            <span style={{ fontSize: 9, color: statusColor, fontFamily: "Space Mono" }}>
                                {demoMode ? "DEMO" : connected ? "LIVE" : "CONNECTING"}
                            </span>
                        </div>
                    </div>

                    {/* Stat cards */}
                    <div style={{ display: "flex", gap: 14, marginBottom: 20, flexWrap: "wrap" }}>
                        <StatCard label="Total Vehicles" value={total} accent="#00d4ff" sub="session total" />
                        <StatCard label="Inbound" value={inCount} accent="#34d399" sub="crossing into frame" />
                        <StatCard label="Outbound" value={outCount} accent="#fb7185" sub="crossing out of frame" />
                        <StatCard label="Active Classes" value={activeClasses} accent="#ffd700" sub={"of " + VEHICLE_TYPES.length + " classes"} />
                    </div>

                    {/* Main grid */}
                    <div style={{
                        display: "grid", gridTemplateColumns: "minmax(0,1fr) 360px",
                        gap: 20, alignItems: "start"
                    }}>

                        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                            {/* VIDEO PANEL */}
                            <div style={{
                                background: "rgba(255,255,255,0.015)",
                                border: "1px solid rgba(255,255,255,0.05)", borderRadius: 14,
                                overflow: "hidden", aspectRatio: "16/9", position: "relative",
                                display: "flex", alignItems: "center", justifyContent: "center"
                            }}>

                                {/* Webcam MJPEG stream */}
                                {webcamMode && webcamOn && (
                                    <img
                                        src={localUrlRef.current + "/video_feed"}
                                        alt="Live webcam feed"
                                        style={{
                                            position: "absolute", inset: 0, width: "100%",
                                            height: "100%", objectFit: "cover"
                                        }}
                                    />
                                )}

                                {/* Processed video (HTML5) */}
                                {!webcamMode && (
                                    <video ref={videoRef} controls muted style={{
                                        position: "absolute", inset: 0, width: "100%", height: "100%",
                                        objectFit: "cover",
                                        display: procState === "done" ? "block" : "none"
                                    }} />
                                )}

                                {/* Play button fallback */}
                                {!webcamMode && procState === "done" && (
                                    <div style={{
                                        position: "absolute", bottom: 56, left: "50%",
                                        transform: "translateX(-50%)", zIndex: 10
                                    }}>
                                        <button onClick={() => videoRef.current && videoRef.current.play()}
                                            style={{
                                                background: "rgba(0,212,255,0.15)",
                                                border: "1px solid rgba(0,212,255,0.4)", borderRadius: 8,
                                                color: "#00d4ff", padding: "7px 18px", fontSize: 9,
                                                cursor: "pointer", fontFamily: "Space Mono", letterSpacing: 2,
                                                backdropFilter: "blur(8px)"
                                            }}>
                                            PLAY VIDEO
                                        </button>
                                    </div>
                                )}

                                {/* Processing overlay */}
                                {!webcamMode && procState !== "done" && (
                                    <div style={{ textAlign: "center", padding: 32 }}>
                                        {(procState === "processing" || procState === "reencoding") ? (
                                            <div>
                                                <div style={{
                                                    width: 48, height: 48,
                                                    border: "3px solid rgba(0,212,255,0.1)",
                                                    borderTop: "3px solid #00d4ff", borderRadius: "50%",
                                                    animation: "spin 1s linear infinite",
                                                    margin: "0 auto 20px"
                                                }} />
                                                <div style={{
                                                    fontFamily: "Space Mono", fontSize: 11,
                                                    color: "#00d4ff", marginBottom: 16, letterSpacing: 1
                                                }}>
                                                    {procState === "reencoding"
                                                        ? "RE-ENCODING TO H.264..."
                                                        : "PROCESSING VIDEO - " + procPct + "%"}
                                                </div>
                                                <div style={{
                                                    width: 280, height: 4,
                                                    background: "rgba(255,255,255,0.06)", borderRadius: 2,
                                                    overflow: "hidden", margin: "0 auto 10px"
                                                }}>
                                                    <div style={{
                                                        width: (procState === "reencoding" ? "100" : procPct) + "%",
                                                        height: "100%",
                                                        background: "linear-gradient(90deg,#00d4ff,#0066ff)",
                                                        borderRadius: 2, transition: "width 0.5s ease"
                                                    }} />
                                                </div>
                                                <div style={{
                                                    fontFamily: "Space Mono",
                                                    fontSize: 9, color: "#334155"
                                                }}>{procMsg}</div>
                                                <div style={{
                                                    fontFamily: "Space Mono",
                                                    fontSize: 8, color: "#1e3a5f", marginTop: 8
                                                }}>
                                                    YOLOv8s inference on RTX 3070
                                                </div>
                                            </div>
                                        ) : procState === "error" ? (
                                            <div>
                                                <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.4 }}>!</div>
                                                <div style={{
                                                    fontFamily: "Space Mono",
                                                    fontSize: 10, color: "#fb7185"
                                                }}>{procMsg}</div>
                                            </div>
                                        ) : (
                                            <div>
                                                <div style={{ fontSize: 44, marginBottom: 12, opacity: 0.1 }}>
                                                    {webcamMode ? "O" : "[]"}
                                                </div>
                                                <div style={{
                                                    fontFamily: "Space Mono", fontSize: 10,
                                                    color: "#1e3a5f", letterSpacing: 2
                                                }}>
                                                    {webcamMode ? "STARTING WEBCAM..." : "UPLOAD A VIDEO TO BEGIN"}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Corner brackets */}
                                {["tl", "tr", "bl", "br"].map(pos => (
                                    <div key={pos} style={{
                                        position: "absolute",
                                        top: pos.startsWith("t") ? 12 : "auto",
                                        bottom: pos.startsWith("b") ? 12 : "auto",
                                        left: pos.endsWith("l") ? 12 : "auto",
                                        right: pos.endsWith("r") ? 12 : "auto",
                                        width: 20, height: 20, pointerEvents: "none",
                                        borderTop: pos.startsWith("t") ? "1px solid rgba(0,212,255,0.4)" : "none",
                                        borderBottom: pos.startsWith("b") ? "1px solid rgba(0,212,255,0.4)" : "none",
                                        borderLeft: pos.endsWith("l") ? "1px solid rgba(0,212,255,0.4)" : "none",
                                        borderRight: pos.endsWith("r") ? "1px solid rgba(0,212,255,0.4)" : "none"
                                    }} />
                                ))}

                                {/* HUD badges */}
                                <div style={{
                                    position: "absolute", top: 14, left: 14,
                                    zIndex: 5, pointerEvents: "none"
                                }}>
                                    <div style={{
                                        display: "flex", alignItems: "center", gap: 5,
                                        background: "rgba(0,0,0,0.7)", borderRadius: 4,
                                        padding: "3px 10px",
                                        border: "1px solid rgba(0,212,255,0.2)"
                                    }}>
                                        <div style={{
                                            width: 5, height: 5, borderRadius: "50%",
                                            background: (procState === "done" || webcamOn) ? "#ef4444" : "#334155",
                                            animation: (procState === "done" || webcamOn)
                                                ? "pulseGlow 1s infinite" : "none"
                                        }} />
                                        <span style={{
                                            fontFamily: "Space Mono", fontSize: 8,
                                            color: "#94a3b8", letterSpacing: 2
                                        }}>
                                            {webcamMode ? "LIVE WEBCAM" : "ROXAS BLVD - KALAW"}
                                        </span>
                                    </div>
                                </div>
                                <div style={{
                                    position: "absolute", top: 14, right: 14,
                                    display: "flex", gap: 6, zIndex: 5, pointerEvents: "none"
                                }}>
                                    <div style={{
                                        background: "rgba(0,0,0,0.7)",
                                        border: "1px solid rgba(255,215,0,0.2)", borderRadius: 4,
                                        padding: "3px 9px"
                                    }}>
                                        <span style={{
                                            fontFamily: "Space Mono", fontSize: 8,
                                            color: "#ffd700", letterSpacing: 2
                                        }}>OC-SORT</span>
                                    </div>
                                    <div style={{
                                        background: "rgba(0,0,0,0.7)",
                                        border: "1px solid rgba(0,212,255,0.12)", borderRadius: 4,
                                        padding: "3px 9px"
                                    }}>
                                        <span style={{
                                            fontFamily: "Space Mono", fontSize: 8,
                                            color: "#475569", letterSpacing: 2
                                        }}>YOLOv8s</span>
                                    </div>
                                </div>
                            </div>

                            {/* Ablation panel */}
                            <div style={{
                                background: "rgba(255,255,255,0.015)",
                                border: "1px solid rgba(255,255,255,0.05)",
                                borderRadius: 12, padding: "18px 20px"
                            }}>
                                <div style={{
                                    display: "flex", justifyContent: "space-between",
                                    alignItems: "center", marginBottom: 14
                                }}>
                                    <div>
                                        <div style={{
                                            fontSize: 9, color: "#475569",
                                            fontFamily: "Space Mono", letterSpacing: 2
                                        }}>
                                            PER-CLASS AVERAGE PRECISION
                                        </div>
                                        <div style={{
                                            fontSize: 8, color: "#1e3a5f",
                                            fontFamily: "Space Mono", marginTop: 2
                                        }}>
                                            offline ablation - finetuned model - test set
                                        </div>
                                    </div>
                                    <div style={{
                                        fontSize: 8, color: "#334155", fontFamily: "Space Mono",
                                        background: "rgba(255,255,255,0.03)", padding: "3px 8px",
                                        borderRadius: 4
                                    }}>
                                        AP = offline - CER needs human counts
                                    </div>
                                </div>
                                <AblationPanel />
                            </div>
                        </div>

                        {/* RIGHT */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                            {/* Detection log */}
                            <div style={{
                                background: "rgba(255,255,255,0.015)",
                                border: "1px solid rgba(255,255,255,0.05)",
                                borderRadius: 12, padding: "18px"
                            }}>
                                <div style={{
                                    display: "flex", justifyContent: "space-between",
                                    alignItems: "center", marginBottom: 12
                                }}>
                                    <div style={{
                                        fontSize: 9, color: "#475569",
                                        fontFamily: "Space Mono", letterSpacing: 2
                                    }}>DETECTION LOG</div>
                                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                                        <div style={{
                                            width: 5, height: 5, borderRadius: "50%",
                                            background: (procState === "processing" || webcamOn)
                                                ? "#00d4ff" : procState === "done" ? "#34d399" : "#334155",
                                            animation: (procState === "processing" || webcamOn || procState === "done")
                                                ? "pulseGlow 1.5s infinite" : "none"
                                        }} />
                                        <span style={{
                                            fontSize: 8, fontFamily: "Space Mono",
                                            color: (procState === "processing" || webcamOn)
                                                ? "#00d4ff" : procState === "done" ? "#34d399" : "#334155"
                                        }}>
                                            {webcamOn ? "LIVE" : procState === "processing" ? "DETECTING"
                                                : procState === "done" ? "COMPLETE" : "IDLE"}
                                        </span>
                                    </div>
                                </div>
                                <div style={{
                                    display: "grid", gridTemplateColumns: "1fr auto auto auto",
                                    gap: "3px 8px", padding: "3px 8px",
                                    borderBottom: "1px solid rgba(255,255,255,0.04)", marginBottom: 6
                                }}>
                                    {["CLASS", "DIR", "TIME", "CONF"].map(h => (
                                        <div key={h} style={{
                                            fontSize: 7, color: "#1e3a5f",
                                            fontFamily: "Space Mono", letterSpacing: 2
                                        }}>{h}</div>
                                    ))}
                                </div>
                                <div style={{
                                    display: "flex", flexDirection: "column", gap: 2,
                                    maxHeight: 340, overflowY: "auto"
                                }}>
                                    {log.length === 0 ? (
                                        <div style={{
                                            textAlign: "center", padding: 40, color: "#1e3a5f",
                                            fontSize: 9, fontFamily: "Space Mono"
                                        }}>
                                            {webcamOn ? "DETECTING VEHICLES..." : "AWAITING UPLOAD..."}
                                        </div>
                                    ) : log.map((entry, i) => {
                                        const col = CLASS_COLORS[entry.vehicleType] || "#64748b";
                                        return (
                                            <div key={entry.timestamp + "-" + i} style={{
                                                display: "grid", gridTemplateColumns: "1fr auto auto auto",
                                                gap: "3px 8px", alignItems: "center", padding: "6px 8px",
                                                borderRadius: 5,
                                                background: i === 0 ? "rgba(0,212,255,0.04)" : "transparent",
                                                borderLeft: i === 0 ? "2px solid rgba(0,212,255,0.3)"
                                                    : "2px solid transparent",
                                                animation: i === 0 ? "slideIn 0.25s ease" : "none"
                                            }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                                                    <div style={{
                                                        width: 6, height: 6, borderRadius: 1,
                                                        background: col, flexShrink: 0
                                                    }} />
                                                    <span style={{
                                                        fontSize: 10, color: col,
                                                        fontFamily: "Space Mono", fontWeight: 700
                                                    }}>
                                                        {entry.vehicleType}
                                                    </span>
                                                </div>
                                                <span style={{
                                                    fontSize: 9, fontWeight: 700,
                                                    color: entry.direction === "IN" ? "#34d399" : "#fb7185",
                                                    fontFamily: "Space Mono"
                                                }}>{entry.direction}</span>
                                                <span style={{
                                                    fontSize: 9, color: "#334155",
                                                    fontFamily: "Space Mono"
                                                }}>
                                                    {new Date(entry.timestamp).toLocaleTimeString("en-PH",
                                                        {
                                                            hour12: false, hour: "2-digit",
                                                            minute: "2-digit", second: "2-digit"
                                                        })}
                                                </span>
                                                <span style={{
                                                    fontSize: 9, color: "#1e3a5f",
                                                    fontFamily: "Space Mono",
                                                    background: "rgba(255,255,255,0.03)",
                                                    padding: "1px 5px", borderRadius: 3
                                                }}>
                                                    {Math.round(entry.confidence * 100)}%
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Session summary */}
                            <div style={{
                                background: "rgba(255,255,255,0.015)",
                                border: "1px solid rgba(255,255,255,0.05)",
                                borderRadius: 12, padding: "16px 18px"
                            }}>
                                <div style={{
                                    fontSize: 9, color: "#334155", fontFamily: "Space Mono",
                                    letterSpacing: 2, marginBottom: 12
                                }}>SESSION SUMMARY</div>
                                {Object.entries(counts).length === 0 ? (
                                    <div style={{
                                        fontSize: 9, color: "#1e3a5f",
                                        fontFamily: "Space Mono", textAlign: "center", padding: 14
                                    }}>
                                        No data yet
                                    </div>
                                ) : Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([cls, cnt]) => {
                                    const max = Math.max(...Object.values(counts));
                                    return (
                                        <div key={cls} style={{
                                            display: "flex", alignItems: "center",
                                            gap: 10, marginBottom: 8
                                        }}>
                                            <div style={{
                                                width: 6, height: 6, borderRadius: "50%",
                                                background: CLASS_COLORS[cls] || "#475569", flexShrink: 0
                                            }} />
                                            <span style={{
                                                fontSize: 9, color: "#475569",
                                                fontFamily: "Space Mono", width: 72
                                            }}>{cls}</span>
                                            <div style={{
                                                flex: 1, height: 4,
                                                background: "rgba(255,255,255,0.04)",
                                                borderRadius: 2, overflow: "hidden"
                                            }}>
                                                <div style={{
                                                    width: ((cnt / max) * 100) + "%", height: "100%",
                                                    background: CLASS_COLORS[cls] || "#475569",
                                                    borderRadius: 2, opacity: 0.75,
                                                    transition: "width 0.5s"
                                                }} />
                                            </div>
                                            <span style={{
                                                fontFamily: "Space Mono", fontSize: 9,
                                                color: "#94a3b8", width: 26, textAlign: "right"
                                            }}>{cnt}</span>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Legend */}
                            <div style={{
                                background: "rgba(255,255,255,0.015)",
                                border: "1px solid rgba(255,255,255,0.05)",
                                borderRadius: 12, padding: "14px 18px"
                            }}>
                                <div style={{
                                    fontSize: 9, color: "#334155", fontFamily: "Space Mono",
                                    letterSpacing: 2, marginBottom: 10
                                }}>CLASS LEGEND</div>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
                                    {VEHICLE_TYPES.map(v => (
                                        <div key={v} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                            <div style={{
                                                width: 7, height: 7, borderRadius: 2,
                                                background: CLASS_COLORS[v] || "#475569", flexShrink: 0
                                            }} />
                                            <span style={{
                                                fontSize: 9, color: "#475569",
                                                fontFamily: "Space Mono"
                                            }}>{v}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </main>

                <footer style={{
                    borderTop: "1px solid rgba(255,255,255,0.03)",
                    padding: "10px 28px", display: "flex",
                    justifyContent: "space-between", marginTop: 8
                }}>
                    <div style={{ fontSize: 8, color: "#0f2337", fontFamily: "Space Mono" }}>
                        YOLOv8s - OC-SORT - OD3 Dataset Distillation - Minority Quota Synthesis - 10% Domain Bridge
                    </div>
                    <div style={{ fontSize: 8, color: "#0f2337", fontFamily: "Space Mono" }}>
                        TIP MANILA - BS COMPUTER SCIENCE - 2025
                    </div>
                </footer>
            </div>
        </div>
    );
}