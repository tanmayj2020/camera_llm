"use client";

import { useFleetStatus } from "@/lib/hooks";

interface Device {
  device_id: string;
  camera_ids: string[];
  ip_address: string;
  last_heartbeat: number;
  model_version: string;
  cpu_usage: number;
  memory_usage: number;
  fps: number;
  status: string;
}

export default function FleetPage() {
  const { data: devices = [], isLoading } = useFleetStatus(5000) as { data: Device[]; isLoading: boolean };

  const online = devices.filter((d) => d.status === "online").length;
  const offline = devices.length - online;

  const ago = (ts: number) => {
    const s = Math.floor(Date.now() / 1000 - ts);
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center gap-4 mb-6">
        <h2 className="text-lg font-semibold">Edge Fleet</h2>
        <span className="text-sm text-[var(--success)]">{online} online</span>
        {offline > 0 && <span className="text-sm text-[var(--danger)]">{offline} offline</span>}
        <span className="text-sm text-[var(--text-muted)]">{devices.length} total</span>
      </div>
      {devices.length === 0 && <p className="text-sm text-[var(--text-muted)]">No edge devices registered.</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {devices.map((d) => (
          <div key={d.device_id} className={`p-4 rounded-lg border ${d.status === "offline" ? "border-[var(--danger)] bg-red-950/30" : "border-[var(--border)] bg-[var(--surface)]"}`}>
            <div className="flex justify-between items-center mb-3">
              <span className="font-medium">{d.device_id}</span>
              <span className={`text-xs px-2 py-0.5 rounded ${d.status === "online" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
                {d.status}
              </span>
            </div>
            <div className="text-xs text-[var(--text-muted)] space-y-1">
              <p>IP: {d.ip_address || "—"} · Model: {d.model_version || "—"}</p>
              <p>Cameras: {d.camera_ids.length} · FPS: {d.fps.toFixed(1)}</p>
              <p>Heartbeat: {d.last_heartbeat ? ago(d.last_heartbeat) : "never"}</p>
            </div>
            <div className="mt-3 space-y-2">
              <div>
                <div className="flex justify-between text-xs mb-1"><span>CPU</span><span>{d.cpu_usage.toFixed(0)}%</span></div>
                <div className="h-2 bg-[var(--border)] rounded overflow-hidden">
                  <div className={`h-full rounded ${d.cpu_usage > 80 ? "bg-[var(--danger)]" : "bg-[var(--accent)]"}`} style={{ width: `${d.cpu_usage}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1"><span>Memory</span><span>{d.memory_usage.toFixed(0)}%</span></div>
                <div className="h-2 bg-[var(--border)] rounded overflow-hidden">
                  <div className={`h-full rounded ${d.memory_usage > 80 ? "bg-[var(--warning)]" : "bg-[var(--accent)]"}`} style={{ width: `${d.memory_usage}%` }} />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
