"use client";

import { useState, useEffect } from "react";
import { fetchAPI } from "../../lib/api";

interface Tracker {
  tracker_id: string;
  name: string;
  description: string;
  severity: string;
  status: string;
  trigger_count: number;
  created_at: number;
  last_triggered: number | null;
}

const SEVERITY_COLORS: Record<string, string> = {
  low: "bg-blue-600",
  medium: "bg-yellow-600",
  high: "bg-orange-600",
  critical: "bg-red-600",
};

export default function TrackersPage() {
  const [trackers, setTrackers] = useState<Tracker[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const load = () => fetchAPI("/api/trackers").then(setTrackers).catch(console.error);

  useEffect(() => { load(); }, []);

  const create = async () => {
    if (!input.trim()) return;
    setLoading(true);
    try {
      await fetchAPI("/api/trackers", {
        method: "POST",
        body: JSON.stringify({ description: input }),
      });
      setInput("");
      await load();
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const toggle = async (id: string, current: string) => {
    const status = current === "active" ? "paused" : "active";
    await fetchAPI(`/api/trackers/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    });
    await load();
  };

  const remove = async (id: string) => {
    await fetchAPI(`/api/trackers/${id}`, { method: "DELETE" });
    await load();
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Custom Trackers</h1>
      <p className="text-[var(--text-muted)] text-sm">
        Describe what to track in plain English. The system will parse it into detection rules automatically.
      </p>

      <div className="flex gap-2">
        <input
          className="flex-1 bg-[var(--card)] border border-[var(--border)] rounded px-3 py-2 text-white text-sm"
          placeholder='e.g. "Alert me when someone enters parking lot after 10pm"'
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && create()}
        />
        <button
          onClick={create}
          disabled={loading}
          className="bg-[var(--accent)] text-black px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
        >
          {loading ? "Creating..." : "Create"}
        </button>
      </div>

      <div className="space-y-3">
        {trackers.length === 0 && (
          <p className="text-[var(--text-muted)] text-sm text-center py-8">No trackers yet. Create one above.</p>
        )}
        {trackers.map((t) => (
          <div key={t.tracker_id} className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-white font-medium text-sm">{t.name}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${SEVERITY_COLORS[t.severity] || "bg-gray-600"} text-white`}>
                  {t.severity}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded ${t.status === "active" ? "bg-green-700" : "bg-gray-600"} text-white`}>
                  {t.status}
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => toggle(t.tracker_id, t.status)}
                  className="text-xs text-[var(--text-muted)] hover:text-white border border-[var(--border)] px-2 py-1 rounded"
                >
                  {t.status === "active" ? "Pause" : "Resume"}
                </button>
                <button
                  onClick={() => remove(t.tracker_id)}
                  className="text-xs text-red-400 hover:text-red-300 border border-[var(--border)] px-2 py-1 rounded"
                >
                  Delete
                </button>
              </div>
            </div>
            <p className="text-[var(--text-muted)] text-xs mb-1">{t.description}</p>
            <div className="flex gap-4 text-xs text-[var(--text-muted)]">
              <span>Triggers: {t.trigger_count}</span>
              {t.last_triggered && (
                <span>Last: {new Date(t.last_triggered * 1000).toLocaleString()}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
