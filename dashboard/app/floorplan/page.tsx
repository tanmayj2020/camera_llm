"use client";

import { useEffect, useState } from "react";
import { fetchAPI } from "../../lib/api";

interface FloorPlanData {
  timestamp: number;
  site: { width: number; height: number };
  stats: Record<string, number>;
  entities: { track_id: string; class: string; x: number; y: number; vx: number; vy: number; camera: string }[];
  cameras: { id: string; x: number; y: number; rotation: number; fov: number; label: string }[];
  zones: { id: string; name: string; polygon: { x: number; y: number }[]; occupancy: number }[];
  alerts: { id: string; x: number; y: number; severity: string; description: string; camera: string }[];
  heatmap: number[][];
}

export default function FloorPlanPage() {
  const [data, setData] = useState<FloorPlanData | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const load = () => fetchAPI("/api/floorplan").then(setData).catch(console.error);

  useEffect(() => {
    load();
    const interval = setInterval(load, 3000);
    return () => clearInterval(interval);
  }, []);

  if (!data) return <p className="text-[var(--text-muted)]">Loading floor plan...</p>;

  const W = data.site.width;
  const H = data.site.height;
  const scale = (v: number, max: number) => (v / max) * 100;

  const severityColor: Record<string, string> = {
    critical: "#ef4444", high: "#f97316", medium: "#eab308", low: "#6b7280",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">🏗️ Digital Twin — Floor Plan</h1>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
            Heatmap
          </label>
          <div className="flex gap-3 text-xs text-[var(--text-muted)]">
            <span>👤 {data.stats.person_count} people</span>
            <span>🚗 {data.stats.vehicle_count} vehicles</span>
            <span>📹 {data.stats.active_cameras} cameras</span>
            <span>🚨 {data.stats.active_alerts} alerts</span>
          </div>
        </div>
      </div>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
        <svg viewBox={`0 0 100 100`} className="w-full" style={{ maxHeight: "70vh" }}>
          <rect x="0" y="0" width="100" height="100" fill="var(--bg)" stroke="var(--border)" strokeWidth="0.3" />

          {/* Grid */}
          {Array.from({ length: 10 }, (_, i) => (
            <g key={`grid-${i}`}>
              <line x1={i * 10} y1="0" x2={i * 10} y2="100" stroke="var(--border)" strokeWidth="0.15" />
              <line x1="0" y1={i * 10} x2="100" y2={i * 10} stroke="var(--border)" strokeWidth="0.15" />
            </g>
          ))}

          {/* Heatmap overlay */}
          {showHeatmap && data.heatmap.map((row, gy) =>
            row.map((val, gx) => val > 0.05 ? (
              <rect key={`hm-${gy}-${gx}`}
                x={gx * (100 / row.length)} y={gy * (100 / data.heatmap.length)}
                width={100 / row.length} height={100 / data.heatmap.length}
                fill={`rgba(59,130,246,${Math.min(val * 0.8, 0.6)})`} />
            ) : null)
          )}

          {/* Zones */}
          {data.zones.map((z) => {
            if (!z.polygon.length) return null;
            const points = z.polygon.map(p => `${scale(p.x, W)},${scale(p.y, H)}`).join(" ");
            return (
              <g key={`zone-${z.id}`}>
                <polygon points={points} fill="rgba(59,130,246,0.08)" stroke="var(--accent)" strokeWidth="0.3" strokeDasharray="1" />
                <text x={scale(z.polygon[0].x, W) + 1} y={scale(z.polygon[0].y, H) + 3}
                  fontSize="2.5" fill="var(--text-muted)">{z.name} ({z.occupancy})</text>
              </g>
            );
          })}

          {/* Cameras */}
          {data.cameras.map((cam) => {
            const cx = scale(cam.x, W);
            const cy = scale(cam.y, H);
            // FOV cone
            const fovRad = (cam.fov / 2) * Math.PI / 180;
            const rotRad = cam.rotation * Math.PI / 180;
            const len = 12;
            const lx = cx + Math.sin(rotRad - fovRad) * len;
            const ly = cy - Math.cos(rotRad - fovRad) * len;
            const rx = cx + Math.sin(rotRad + fovRad) * len;
            const ry = cy - Math.cos(rotRad + fovRad) * len;
            return (
              <g key={`cam-${cam.id}`}>
                <polygon points={`${cx},${cy} ${lx},${ly} ${rx},${ry}`}
                  fill="rgba(59,130,246,0.06)" stroke="var(--accent)" strokeWidth="0.2" />
                <circle cx={cx} cy={cy} r="1.5" fill="var(--accent)" />
                <text x={cx} y={cy - 2.5} fontSize="2.2" fill="var(--accent)" textAnchor="middle">{cam.label}</text>
              </g>
            );
          })}

          {/* Entities */}
          {data.entities.map((e) => {
            const ex = scale(e.x, W);
            const ey = scale(e.y, H);
            const isPerson = e.class === "person";
            const isSelected = selected === e.track_id;
            return (
              <g key={`ent-${e.track_id}`} onClick={() => setSelected(e.track_id)} style={{ cursor: "pointer" }}>
                <circle cx={ex} cy={ey} r={isSelected ? 2.5 : 1.8}
                  fill={isPerson ? "var(--accent)" : "var(--warning)"}
                  stroke={isSelected ? "white" : "none"} strokeWidth="0.4" />
                {/* Velocity arrow */}
                {(Math.abs(e.vx) > 0.1 || Math.abs(e.vy) > 0.1) && (
                  <line x1={ex} y1={ey}
                    x2={ex + e.vx * 2} y2={ey + e.vy * 2}
                    stroke={isPerson ? "var(--accent)" : "var(--warning)"} strokeWidth="0.3"
                    markerEnd="url(#arrow)" />
                )}
                <text x={ex} y={ey - 2.5} fontSize="2" fill="var(--text)" textAnchor="middle">
                  {isPerson ? "👤" : "🚗"}
                </text>
              </g>
            );
          })}

          {/* Alerts */}
          {data.alerts.map((a) => {
            const ax = scale(a.x, W);
            const ay = scale(a.y, H);
            return (
              <g key={`alert-${a.id}`}>
                <circle cx={ax} cy={ay} r="3" fill="none"
                  stroke={severityColor[a.severity] || "#888"} strokeWidth="0.4">
                  <animate attributeName="r" values="2;4;2" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite" />
                </circle>
                <text x={ax} y={ay + 1} fontSize="3" textAnchor="middle">🚨</text>
              </g>
            );
          })}

          <defs>
            <marker id="arrow" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto">
              <path d="M0,0 L4,2 L0,4 Z" fill="var(--text-muted)" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Selected entity detail */}
      {selected && (() => {
        const ent = data.entities.find(e => e.track_id === selected);
        if (!ent) return null;
        return (
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 text-sm">
            <div className="flex justify-between items-center">
              <span className="font-mono text-white">Entity: {ent.track_id}</span>
              <button onClick={() => setSelected(null)} className="text-xs text-[var(--text-muted)]">✕</button>
            </div>
            <div className="grid grid-cols-4 gap-4 mt-2 text-xs text-[var(--text-muted)]">
              <span>Class: {ent.class}</span>
              <span>Camera: {ent.camera}</span>
              <span>Position: ({ent.x}, {ent.y})</span>
              <span>Velocity: ({ent.vx}, {ent.vy})</span>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
