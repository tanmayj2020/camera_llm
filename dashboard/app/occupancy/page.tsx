"use client";

import { useState } from "react";
import { useSiteOccupancy, useOccupancy } from "@/lib/hooks";

interface CameraOccupancy {
  camera_id: string;
  name: string;
  count: number;
  trend: number;
}

interface Zone {
  name: string;
  count: number;
  capacity: number;
}

interface SiteData {
  total_count: number;
  cameras: CameraOccupancy[];
  hourly: number[];
  zones: Zone[];
  summary: string;
}

export default function OccupancyPage() {
  const { data: site, isLoading } = useSiteOccupancy(5000) as { data: SiteData | undefined; isLoading: boolean };
  const [selectedCamera, setSelectedCamera] = useState<string>("");
  const { data: cameraDetail } = useOccupancy(selectedCamera, 5000);

  if (isLoading || !site) return <div className="max-w-4xl mx-auto text-[var(--text-muted)]">Loading…</div>;

  const maxHourly = Math.max(...(site.hourly || []), 1);

  return (
    <div className="max-w-4xl mx-auto">
      {site.summary && (
        <p className="text-sm text-[var(--text-muted)] mb-4 p-3 bg-[var(--surface)] rounded border border-[var(--border)]">{site.summary}</p>
      )}

      <div className="text-center mb-6">
        <div className="text-6xl font-bold text-[var(--accent)]">{site.total_count}</div>
        <div className="text-sm text-[var(--text-muted)]">People on site</div>
      </div>

      <h2 className="text-lg font-semibold mb-2">Per-Camera</h2>
      <div className="grid grid-cols-2 gap-3 mb-6">
        {site.cameras?.map((c) => (
          <div key={c.camera_id} onClick={() => setSelectedCamera(c.camera_id)}
            className={`p-3 rounded border cursor-pointer ${selectedCamera === c.camera_id ? "border-[var(--accent)]" : "border-[var(--border)]"} bg-[var(--surface)]`}>
            <div className="text-sm font-medium text-[var(--text)]">{c.name}</div>
            <div className="text-2xl font-bold text-[var(--text)]">
              {c.count} <span className="text-sm">{c.trend > 0 ? "↑" : c.trend < 0 ? "↓" : "→"}</span>
            </div>
          </div>
        ))}
      </div>

      <h2 className="text-lg font-semibold mb-2">24-Hour Trend</h2>
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4 mb-6">
        <svg viewBox="0 0 480 120" className="w-full">
          {(site.hourly || []).map((v, i) => (
            <g key={i}>
              <rect x={i * 20} y={100 - (v / maxHourly) * 100} width="16" height={(v / maxHourly) * 100}
                fill="var(--accent)" rx="2" />
              <text x={i * 20 + 8} y="115" textAnchor="middle" fontSize="6" fill="var(--text-muted)">{i}</text>
            </g>
          ))}
        </svg>
      </div>

      <h2 className="text-lg font-semibold mb-2">Zone Occupancy</h2>
      <div className="space-y-2">
        {site.zones?.map((z) => {
          const pct = z.capacity > 0 ? (z.count / z.capacity) * 100 : 0;
          const color = pct > 90 ? "var(--danger)" : pct > 70 ? "var(--warning)" : "var(--success)";
          return (
            <div key={z.name} className="bg-[var(--surface)] border border-[var(--border)] rounded p-3">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-[var(--text)]">{z.name}</span>
                <span className="text-[var(--text-muted)]">{z.count}/{z.capacity}</span>
              </div>
              <div className="h-2 bg-[var(--bg)] rounded overflow-hidden">
                <div className="h-full rounded" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
