"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Event { event_id: string; event_type: string; description: string; timestamp: number; camera_id: string; severity: string }

const SEV_COLORS: Record<string, string> = { critical: "#ef4444", high: "#f97316", medium: "#eab308", low: "#6b7280" };

export default function TimelinePage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [camId, setCamId] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [selected, setSelected] = useState<Event | null>(null);
  const [filters, setFilters] = useState<Record<string, boolean>>({ critical: true, high: true, medium: true, low: true });

  useEffect(() => { api.listCameras().then(setCameras).catch(console.error); }, []);

  useEffect(() => {
    if (!camId) return;
    api.getTimeline(camId).then((d: any) => setEvents(Array.isArray(d) ? d : [])).catch(console.error);
  }, [camId]);

  const now = Date.now() / 1000;
  const dayAgo = now - 86400;
  const filtered = events.filter((e) => filters[e.severity]);

  const xPos = (ts: number) => Math.max(0, Math.min(100, ((ts - dayAgo) / 86400) * 100));

  return (
    <div className="max-w-5xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Incident Timeline</h2>

      <div className="flex items-center gap-4 mb-4">
        <select value={camId} onChange={(e) => { setCamId(e.target.value); setSelected(null); }}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]">
          <option value="">Select camera</option>
          {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <div className="flex gap-3">
          {Object.keys(SEV_COLORS).map((s) => (
            <label key={s} className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
              <input type="checkbox" checked={filters[s]} onChange={() => setFilters((p) => ({ ...p, [s]: !p[s] }))} />
              <span style={{ color: SEV_COLORS[s] }}>{s}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 mb-4">
        <div className="flex justify-between text-xs text-[var(--text-muted)] mb-2">
          <span>24h ago</span><span>Now</span>
        </div>
        <svg width="100%" height="40" className="overflow-visible">
          <rect x="0" y="16" width="100%" height="8" rx="4" fill="var(--border)" />
          {filtered.map((e, i) => (
            <circle
              key={i}
              cx={`${xPos(e.timestamp)}%`}
              cy="20"
              r="6"
              fill={SEV_COLORS[e.severity] || SEV_COLORS.low}
              className="cursor-pointer hover:opacity-80"
              onClick={() => setSelected(e)}
            />
          ))}
        </svg>
      </div>

      {selected && (
        <div className="p-4 rounded-lg bg-[var(--surface)] border border-[var(--border)]">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-bold uppercase" style={{ color: SEV_COLORS[selected.severity] }}>{selected.severity}</span>
            <span className="text-xs text-[var(--text-muted)]">{selected.event_type}</span>
            <span className="text-xs text-[var(--text-muted)]">{selected.camera_id}</span>
          </div>
          <p className="text-sm mb-1">{selected.description}</p>
          <p className="text-xs text-[var(--text-muted)]">{new Date(selected.timestamp * 1000).toLocaleString()}</p>
        </div>
      )}

      {camId && events.length === 0 && <p className="text-sm text-[var(--text-muted)]">No events in the last 24 hours.</p>}
    </div>
  );
}
