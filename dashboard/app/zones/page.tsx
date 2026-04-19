"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Zone { zone_id: string; name: string; type: string; polygon: any; capacity?: number }

export default function ZonesPage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [camId, setCamId] = useState("");
  const [zones, setZones] = useState<Zone[]>([]);
  const [entities, setEntities] = useState<any[]>([]);
  const [form, setForm] = useState({ zone_id: "", name: "", type: "monitored", capacity: "", x1: "", y1: "", x2: "", y2: "" });

  useEffect(() => { api.listCameras().then(setCameras).catch(console.error); }, []);

  useEffect(() => {
    if (!camId) return;
    api.getSpatialState(camId).then((s: any) => {
      setZones(s.zones || []);
      setEntities(s.entities || []);
    }).catch(console.error);
  }, [camId]);

  const occupancy = (z: Zone) => entities.filter(() => true).length; // simplified

  const addZone = async () => {
    const polygon = [[+form.x1, +form.y1], [+form.x2, +form.y1], [+form.x2, +form.y2], [+form.x1, +form.y2]];
    await api.addZone({ zone_id: form.zone_id, name: form.name, type: form.type, polygon, capacity: form.capacity ? +form.capacity : undefined });
    setForm({ zone_id: "", name: "", type: "monitored", capacity: "", x1: "", y1: "", x2: "", y2: "" });
    if (camId) api.getSpatialState(camId).then((s: any) => setZones(s.zones || []));
  };

  const deleteZone = async (id: string) => {
    await api.deleteZone(id);
    setZones((p) => p.filter((z) => z.zone_id !== id));
  };

  const typeBadge: Record<string, string> = {
    restricted: "bg-red-900 text-red-300",
    monitored: "bg-blue-900 text-blue-300",
    counting: "bg-green-900 text-green-300",
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Zone Management</h2>

      <select value={camId} onChange={(e) => setCamId(e.target.value)}
        className="mb-4 bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]">
        <option value="">Select camera</option>
        {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>

      <div className="space-y-2 mb-6">
        {zones.map((z) => (
          <div key={z.zone_id} className="flex items-center justify-between p-3 rounded-lg bg-[var(--surface)] border border-[var(--border)]">
            <div className="flex items-center gap-3">
              <span className="font-medium text-[var(--text)]">{z.name}</span>
              <span className={`text-xs px-2 py-0.5 rounded ${typeBadge[z.type] || "bg-gray-800"}`}>{z.type}</span>
              <span className="text-xs text-[var(--text-muted)]">Occupancy: {occupancy(z)}{z.capacity ? `/${z.capacity}` : ""}</span>
            </div>
            <button onClick={() => deleteZone(z.zone_id)} className="text-xs text-[var(--danger)] hover:underline">Delete</button>
          </div>
        ))}
        {camId && zones.length === 0 && <p className="text-sm text-[var(--text-muted)]">No zones configured.</p>}
      </div>

      <div className="p-4 rounded-lg bg-[var(--surface)] border border-[var(--border)]">
        <h3 className="text-sm font-semibold mb-3">Add Zone</h3>
        <div className="grid grid-cols-2 gap-3">
          <input placeholder="Zone ID" value={form.zone_id} onChange={(e) => setForm({ ...form, zone_id: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <input placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]">
            <option value="restricted">Restricted</option>
            <option value="monitored">Monitored</option>
            <option value="counting">Counting</option>
          </select>
          <input placeholder="Capacity" type="number" value={form.capacity} onChange={(e) => setForm({ ...form, capacity: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <input placeholder="x1" type="number" value={form.x1} onChange={(e) => setForm({ ...form, x1: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <input placeholder="y1" type="number" value={form.y1} onChange={(e) => setForm({ ...form, y1: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <input placeholder="x2" type="number" value={form.x2} onChange={(e) => setForm({ ...form, x2: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
          <input placeholder="y2" type="number" value={form.y2} onChange={(e) => setForm({ ...form, y2: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-1.5 text-sm text-[var(--text)]" />
        </div>
        <button onClick={addZone} disabled={!form.zone_id || !form.name}
          className="mt-3 bg-[var(--accent)] hover:opacity-90 disabled:opacity-50 px-4 py-1.5 rounded text-sm text-white">
          Add Zone
        </button>
      </div>
    </div>
  );
}
