"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function GraphPage() {
  const [cameraId, setCameraId] = useState("cam-0");
  const [events, setEvents] = useState<any[]>([]);
  const [spatial, setSpatial] = useState<any>(null);

  useEffect(() => {
    api.getRecentGraph(cameraId).then(setEvents).catch(() => {});
    api.getSpatialState(cameraId).then(setSpatial).catch(() => {});
  }, [cameraId]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Knowledge Graph Events */}
      <div>
        <div className="flex items-center gap-4 mb-4">
          <h2 className="text-lg font-semibold">Knowledge Graph</h2>
          <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}
            className="bg-[var(--surface)] border border-[var(--border)] rounded px-2 py-1 text-sm">
            {["cam-0", "cam-1", "cam-2", "cam-3"].map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 max-h-[600px] overflow-y-auto space-y-2">
          {events.length === 0 && <p className="text-sm text-[var(--text-muted)]">No graph data yet.</p>}
          {events.map((evt: any, i: number) => (
            <div key={i} className="text-xs border-b border-[var(--border)] pb-2">
              <span className="text-[var(--text-muted)]">{evt.timestamp}</span>
              <span className="ml-2 font-mono">{evt.event_type}</span>
              <span className="ml-2 text-[var(--text-muted)]">{evt.event_id?.slice(0, 8)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 3D Spatial View */}
      <div>
        <h2 className="text-lg font-semibold mb-4">3D Spatial View</h2>
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 aspect-square relative">
          {/* Simple top-down spatial visualization */}
          <svg viewBox="0 0 100 100" className="w-full h-full">
            <rect x="0" y="0" width="100" height="100" fill="none" stroke="var(--border)" strokeWidth="0.5" />
            {/* Grid */}
            {Array.from({ length: 10 }, (_, i) => (
              <g key={i}>
                <line x1={i * 10} y1="0" x2={i * 10} y2="100" stroke="var(--border)" strokeWidth="0.2" />
                <line x1="0" y1={i * 10} x2="100" y2={i * 10} stroke="var(--border)" strokeWidth="0.2" />
              </g>
            ))}
            {/* Zones */}
            {spatial?.zones?.map((z: any, i: number) => (
              <g key={`z-${i}`}>
                <rect x={20 + i * 25} y={20} width={20} height={20}
                  fill="rgba(59,130,246,0.1)" stroke="var(--accent)" strokeWidth="0.3" />
                <text x={30 + i * 25} y={32} fontSize="3" fill="var(--text-muted)" textAnchor="middle">
                  {z.name}
                </text>
              </g>
            ))}
            {/* Entities */}
            {spatial?.entities?.map((e: any, i: number) => {
              const x = Math.min(95, Math.max(5, (e.position[0] + 10) * 5));
              const y = Math.min(95, Math.max(5, e.position[2] * 5));
              return (
                <g key={`e-${i}`}>
                  <circle cx={x} cy={y} r="2"
                    fill={e.class_name === "person" ? "var(--accent)" : "var(--warning)"} />
                  <text x={x} y={y - 3} fontSize="2.5" fill="var(--text)" textAnchor="middle">
                    {e.track_id}
                  </text>
                </g>
              );
            })}
            {/* Camera icon */}
            <text x="50" y="95" fontSize="4" textAnchor="middle">📹</text>
          </svg>
          <div className="absolute bottom-2 left-2 text-xs text-[var(--text-muted)]">
            {spatial?.entities?.length || 0} entities | {spatial?.zones?.length || 0} zones
          </div>
        </div>
      </div>
    </div>
  );
}
