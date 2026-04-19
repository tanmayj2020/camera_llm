"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Entity { track_id: string; class_name: string }
interface CamState { entities: Entity[]; zones: any[] }

export default function VideoWallPage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [states, setStates] = useState<Record<string, CamState>>({});
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => { api.listCameras().then(setCameras).catch(console.error); }, []);

  useEffect(() => {
    if (!cameras.length) return;
    const fetch = () => cameras.forEach((cam) =>
      api.getSpatialState(cam).then((s: CamState) =>
        setStates((p) => ({ ...p, [cam]: s }))
      ).catch(console.error)
    );
    fetch();
    const id = setInterval(fetch, 2000);
    return () => clearInterval(id);
  }, [cameras]);

  const severity = (s: CamState | undefined) => {
    if (!s) return "bg-gray-500";
    const n = s.entities.length;
    return n > 10 ? "bg-red-500" : n > 5 ? "bg-yellow-500" : "bg-green-500";
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Video Wall</h2>
      <div className="grid grid-cols-2 gap-4">
        {cameras.map((cam) => (
          <div
            key={cam}
            onClick={() => setSelected(selected === cam ? null : cam)}
            className={`p-4 rounded-lg border cursor-pointer transition-colors ${
              selected === cam ? "border-[var(--accent)]" : "border-[var(--border)]"
            } bg-[var(--surface)]`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-[var(--text)]">{cam}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs bg-[var(--accent)] px-2 py-0.5 rounded-full">
                  {states[cam]?.entities.length ?? 0}
                </span>
                <span className={`w-2.5 h-2.5 rounded-full ${severity(states[cam])}`} />
              </div>
            </div>
            <div className="text-xs text-[var(--text-muted)] space-y-0.5">
              {(selected === cam ? states[cam]?.entities : states[cam]?.entities.slice(0, 3))?.map((e) => (
                <div key={e.track_id}>{e.track_id} — {e.class_name}</div>
              ))}
              {!selected && (states[cam]?.entities.length ?? 0) > 3 && (
                <div className="text-[var(--accent)]">+{states[cam]!.entities.length - 3} more</div>
              )}
            </div>
          </div>
        ))}
      </div>
      {cameras.length === 0 && <p className="text-sm text-[var(--text-muted)]">No cameras found.</p>}
    </div>
  );
}
