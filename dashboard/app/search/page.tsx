"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";

interface Result { result_id: string; timestamp: number; camera_id: string; description: string; entity_class: string; relevance: number }

export default function SearchPage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [accessLevel, setAccessLevel] = useState("anonymous");
  const [camFilter, setCamFilter] = useState("");
  const [timeRange, setTimeRange] = useState("24h");
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => { api.listCameras().then(setCameras).catch(console.error); }, []);

  const search = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await api.forensicSearch(query, accessLevel, camFilter || undefined);
      setResults(Array.isArray(res) ? res : res.results || []);
    } catch { setResults([]); }
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Visual AI Search</h2>

      <div className="flex gap-2 mb-4">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder="Search across all cameras... (e.g. 'person with red backpack')"
          className="flex-1 bg-[var(--surface)] border border-[var(--border)] rounded px-4 py-2.5 text-sm text-[var(--text)]"
        />
        <button onClick={search} disabled={loading}
          className="bg-[var(--accent)] hover:opacity-90 disabled:opacity-50 px-5 py-2.5 rounded text-sm text-white">
          {loading ? "…" : "Search"}
        </button>
      </div>

      <div className="flex gap-3 mb-6 flex-wrap">
        <div className="flex gap-1">
          {["1h", "6h", "24h", "7d"].map((t) => (
            <button key={t} onClick={() => setTimeRange(t)}
              className={`text-xs px-3 py-1 rounded ${timeRange === t ? "bg-[var(--accent)] text-white" : "bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)]"}`}>
              {t}
            </button>
          ))}
        </div>
        <select value={camFilter} onChange={(e) => setCamFilter(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1 text-xs text-[var(--text)]">
          <option value="">All cameras</option>
          {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select value={accessLevel} onChange={(e) => setAccessLevel(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1 text-xs text-[var(--text)]">
          <option value="anonymous">Anonymous</option>
          <option value="operator">Operator</option>
          <option value="supervisor">Supervisor</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {results.map((r) => (
          <div key={r.result_id} className="p-3 rounded-lg bg-[var(--surface)] border border-[var(--border)]">
            <div className="flex justify-between items-start mb-1">
              <span className="text-xs text-[var(--text-muted)]">{r.camera_id}</span>
              <span className="text-xs text-[var(--accent)]">{Math.round((r.relevance || 0) * 100)}%</span>
            </div>
            <p className="text-sm mb-1">{r.description}</p>
            <div className="flex gap-2 text-xs text-[var(--text-muted)]">
              <span>{r.entity_class}</span>
              <span>{new Date(r.timestamp * 1000).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>
      {results.length === 0 && !loading && <p className="text-sm text-[var(--text-muted)] text-center mt-8">Enter a query to search across camera footage.</p>}
    </div>
  );
}
