"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";

function fmtDuration(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

export default function JourneyPage() {
  const [entityId, setEntityId] = useState("");
  const [journey, setJourney] = useState<any>(null);
  const [active, setActive] = useState<any[]>([]);

  useEffect(() => {
    const load = () => api.getActiveJourneys().then(setActive).catch(() => {});
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  const track = () => {
    if (entityId.trim()) api.getEntityJourney(entityId).then(setJourney).catch(() => {});
  };

  return (
    <div className="p-6 min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <h1 className="text-2xl font-bold mb-4">Entity Journey</h1>
      <div className="flex gap-2 mb-6">
        <input value={entityId} onChange={e => setEntityId(e.target.value)} onKeyDown={e => e.key === "Enter" && track()}
          placeholder="Entity ID" className="px-3 py-2 rounded border border-[var(--border)] bg-[var(--surface)] flex-1" />
        <button onClick={track} className="px-4 py-2 rounded bg-[var(--accent)] text-white font-medium">Track</button>
      </div>

      <div className="mb-6">
        <h2 className="text-lg font-semibold mb-2">Active Journeys</h2>
        <div className="grid gap-2">
          {active.map((a: any) => (
            <button key={a.entity_id} onClick={() => { setEntityId(a.entity_id); api.getEntityJourney(a.entity_id).then(setJourney); }}
              className="p-3 rounded border border-[var(--border)] bg-[var(--surface)] text-left flex justify-between">
              <span className="font-medium">{a.entity_id}</span>
              <span className="text-[var(--text-muted)] text-sm">
                {a.camera_count} cameras · {a.last_camera} · {a.handoff_count} handoffs
              </span>
            </button>
          ))}
          {active.length === 0 && <p className="text-[var(--text-muted)]">No active journeys</p>}
        </div>
      </div>

      {journey && (
        <>
          <div className="flex gap-4 mb-6 p-4 rounded bg-[var(--surface)] border border-[var(--border)]">
            <div><span className="text-[var(--text-muted)] text-sm">Cameras</span><p className="font-bold">{journey.total_cameras}</p></div>
            <div><span className="text-[var(--text-muted)] text-sm">Duration</span><p className="font-bold">{fmtDuration(journey.duration)}</p></div>
            <div><span className="text-[var(--text-muted)] text-sm">Handoffs</span><p className="font-bold text-[var(--success)]">{journey.handoffs?.length ?? 0}</p></div>
          </div>

          <div className="relative pl-6">
            {journey.path?.map((p: any, i: number) => (
              <div key={i}>
                {i > 0 && journey.handoffs?.[i - 1] && (
                  <div className="ml-2 my-1 text-[var(--accent)] text-sm flex items-center gap-1">↓ handoff</div>
                )}
                <div className="flex items-start gap-3 mb-4">
                  <div className="w-3 h-3 mt-1.5 rounded-full bg-[var(--accent)] shrink-0" />
                  <div className="p-3 rounded bg-[var(--surface)] border border-[var(--border)] flex-1">
                    <div className="font-medium">{p.camera_id}</div>
                    <div className="text-sm text-[var(--text-muted)]">
                      {p.entered} · {fmtDuration(p.duration)} · {p.zone}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
