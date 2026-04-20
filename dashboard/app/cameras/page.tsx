"use client";

import { useState } from "react";
import { useCameras, useAddCamera, useDeleteCamera } from "@/lib/hooks";
import { api } from "@/lib/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

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
  const { data: cameras = [], isLoading } = useCameras();
  const addCamera = useAddCamera();
  const deleteCamera = useDeleteCamera();
  const qc = useQueryClient();
  const toggleCamera = useMutation({
    mutationFn: (c: Camera) => api.updateCamera(c.camera_id, { enabled: !c.enabled }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cameras"] }),
  });
  const [form, setForm] = useState({ camera_id: "", name: "", rtsp_url: "", site_x: "0", site_y: "0", rotation: "0" });

  const add = async () => {
    if (!form.camera_id || !form.name) return;
    await addCamera.mutateAsync({
      ...form,
      site_x: parseFloat(form.site_x),
      site_y: parseFloat(form.site_y),
      rotation: parseFloat(form.rotation),
    });
    setForm({ camera_id: "", name: "", rtsp_url: "", site_x: "0", site_y: "0", rotation: "0" });
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
        <button onClick={add} disabled={addCamera.isPending}
          className="bg-[var(--accent)] text-white px-4 py-2 rounded text-sm hover:opacity-80 disabled:opacity-50">
          {addCamera.isPending ? "Adding…" : "Add Camera"}
        </button>
      </div>

      {isLoading && <p className="text-sm text-[var(--text-muted)]">Loading cameras…</p>}

      <div className="space-y-3">
        {cameras.map((c: Camera) => (
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
                <button onClick={() => toggleCamera.mutate(c)}
                  className={`text-xs px-3 py-1 rounded ${c.enabled ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-400"}`}>
                  {c.enabled ? "Enabled" : "Disabled"}
                </button>
                <button onClick={() => deleteCamera.mutate(c.camera_id)} className="text-xs px-3 py-1 rounded bg-red-900 text-red-300 hover:opacity-80">
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
        {!isLoading && cameras.length === 0 && <p className="text-sm text-[var(--text-muted)]">No cameras configured.</p>}
      </div>
    </div>
  );
}
