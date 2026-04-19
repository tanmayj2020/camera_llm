"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Camera {
  camera_id: string;
  name: string;
  rtsp_url: string;
  site_x: number;
  site_y: number;
  rotation: number;
  enabled: boolean;
}

export default function CamerasPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [form, setForm] = useState({ camera_id: "", name: "", rtsp_url: "", site_x: "0", site_y: "0", rotation: "0" });

  const load = () => api.listCameras().then(setCameras).catch(console.error);
  useEffect(() => { load(); }, []);

  const add = async () => {
    if (!form.camera_id || !form.name) return;
    await api.addCamera({
      ...form,
      site_x: parseFloat(form.site_x),
      site_y: parseFloat(form.site_y),
      rotation: parseFloat(form.rotation),
    });
    setForm({ camera_id: "", name: "", rtsp_url: "", site_x: "0", site_y: "0", rotation: "0" });
    load();
  };

  const toggle = async (c: Camera) => {
    await api.updateCamera(c.camera_id, { enabled: !c.enabled });
    load();
  };

  const remove = async (id: string) => {
    await api.deleteCamera(id);
    load();
  };

  const F = (props: { placeholder: string; field: keyof typeof form; type?: string }) => (
    <input placeholder={props.placeholder} type={props.type || "text"} value={form[props.field]}
      onChange={(e) => setForm({ ...form, [props.field]: e.target.value })}
      className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]" />
  );

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Cameras ({cameras.length})</h2>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4 mb-6">
        <div className="grid grid-cols-3 gap-3 mb-3">
          <F placeholder="Camera ID" field="camera_id" />
          <F placeholder="Name" field="name" />
          <F placeholder="RTSP URL" field="rtsp_url" />
          <F placeholder="Site X" field="site_x" type="number" />
          <F placeholder="Site Y" field="site_y" type="number" />
          <F placeholder="Rotation" field="rotation" type="number" />
        </div>
        <button onClick={add} className="bg-[var(--accent)] text-white px-4 py-2 rounded text-sm hover:opacity-80">
          Add Camera
        </button>
      </div>

      <div className="space-y-3">
        {cameras.map((c) => (
          <div key={c.camera_id} className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="font-medium text-[var(--text)]">{c.name}</div>
                <div className="text-xs text-[var(--text-muted)]">ID: {c.camera_id}</div>
                <div className="text-xs text-[var(--text-muted)]">
                  {c.rtsp_url.length > 50 ? c.rtsp_url.slice(0, 50) + "…" : c.rtsp_url}
                </div>
                <div className="text-xs text-[var(--text-muted)]">Position: ({c.site_x}, {c.site_y}) · Rotation: {c.rotation}°</div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => toggle(c)}
                  className={`text-xs px-3 py-1 rounded ${c.enabled ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-400"}`}>
                  {c.enabled ? "Enabled" : "Disabled"}
                </button>
                <button onClick={() => remove(c.camera_id)} className="text-xs px-3 py-1 rounded bg-red-900 text-red-300 hover:opacity-80">
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
        {cameras.length === 0 && <p className="text-sm text-[var(--text-muted)]">No cameras configured.</p>}
      </div>
    </div>
  );
}
