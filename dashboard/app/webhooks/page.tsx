"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Webhook {
  id: string;
  name: string;
  url: string;
  type: string;
  min_severity: string;
  camera_filter: string[];
  enabled: boolean;
  deliveries: number;
  failures: number;
  last_delivered?: string;
}

const TYPES = ["Generic", "Slack", "PagerDuty", "Teams", "SIEM"];
const SEVERITIES = ["low", "medium", "high", "critical"];

export default function WebhooksPage() {
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [form, setForm] = useState({ name: "", url: "", type: "Generic", min_severity: "low", camera_filter: "" });

  const load = () => api.listWebhooks().then(setWebhooks).catch(console.error);
  useEffect(() => { load(); }, []);

  const create = async () => {
    if (!form.name || !form.url) return;
    await api.createWebhook({
      ...form,
      camera_filter: form.camera_filter ? form.camera_filter.split(",").map((s) => s.trim()) : [],
    });
    setForm({ name: "", url: "", type: "Generic", min_severity: "low", camera_filter: "" });
    load();
  };

  const toggle = async (w: Webhook) => {
    await api.updateWebhook(w.id, { enabled: !w.enabled });
    load();
  };

  const remove = async (id: string) => {
    await api.deleteWebhook(id);
    load();
  };

  const typeBadge: Record<string, string> = {
    Slack: "bg-purple-900 text-purple-300",
    PagerDuty: "bg-green-900 text-green-300",
    Teams: "bg-blue-900 text-blue-300",
    SIEM: "bg-yellow-900 text-yellow-300",
    Generic: "bg-gray-800 text-gray-300",
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Webhooks</h2>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4 mb-6 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <input placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]" />
          <input placeholder="URL" value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]" />
          <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]">
            {TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
          <select value={form.min_severity} onChange={(e) => setForm({ ...form, min_severity: e.target.value })}
            className="bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]">
            {SEVERITIES.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        <input placeholder="Camera filter (comma-separated)" value={form.camera_filter}
          onChange={(e) => setForm({ ...form, camera_filter: e.target.value })}
          className="w-full bg-[var(--bg)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text)]" />
        <button onClick={create} className="bg-[var(--accent)] text-white px-4 py-2 rounded text-sm hover:opacity-80">
          Create Webhook
        </button>
      </div>

      <div className="space-y-3">
        {webhooks.map((w) => (
          <div key={w.id} className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-[var(--text)]">{w.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${typeBadge[w.type] || typeBadge.Generic}`}>{w.type}</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300">{w.min_severity}</span>
                </div>
                <p className="text-xs text-[var(--text-muted)] mb-1">{w.url.length > 60 ? w.url.slice(0, 60) + "…" : w.url}</p>
                <div className="text-xs text-[var(--text-muted)]">
                  {w.deliveries} deliveries · {w.failures} failures
                  {w.last_delivered && ` · Last: ${w.last_delivered}`}
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => toggle(w)}
                  className={`text-xs px-3 py-1 rounded ${w.enabled ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-400"}`}>
                  {w.enabled ? "Enabled" : "Disabled"}
                </button>
                <button onClick={() => remove(w.id)} className="text-xs px-3 py-1 rounded bg-red-900 text-red-300 hover:opacity-80">
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
        {webhooks.length === 0 && <p className="text-sm text-[var(--text-muted)]">No webhooks configured.</p>}
      </div>
    </div>
  );
}
