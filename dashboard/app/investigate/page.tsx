"use client";

import { useState } from "react";
import { fetchAPI } from "../../lib/api";

interface InvestigationReport {
  investigation_id: string;
  subject_entity: string;
  subject_profile: any;
  narrative: string;
  risk_assessment: string;
  recommended_actions: string[];
  timeline: { timestamp: number; camera: string; type: string; description: string }[];
  camera_route: { camera_id: string; transit_time_s?: number }[];
  associates: { entity_id: string; relationship: string; co_occurrences: number; risk_score: number }[];
  anomalies: any[];
}

const RISK_COLORS: Record<string, string> = {
  HIGH: "text-red-400",
  MEDIUM: "text-yellow-400",
  LOW: "text-green-400",
};

export default function InvestigatePage() {
  const [entityId, setEntityId] = useState("");
  const [report, setReport] = useState<InvestigationReport | null>(null);
  const [loading, setLoading] = useState(false);

  const investigate = async () => {
    if (!entityId.trim()) return;
    setLoading(true);
    try {
      const data = await fetchAPI(`/api/investigate/${entityId}`, { method: "POST" });
      setReport(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const riskColor = report?.risk_assessment
    ? RISK_COLORS[Object.keys(RISK_COLORS).find(k => report.risk_assessment.includes(k)) || "LOW"]
    : "";

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">🔍 Autonomous Investigation</h1>
      <p className="text-[var(--text-muted)] text-sm">
        Enter an entity/track ID to auto-investigate: traces across cameras, finds associates, generates report.
      </p>

      <div className="flex gap-2">
        <input
          className="flex-1 bg-[var(--card)] border border-[var(--border)] rounded px-3 py-2 text-white text-sm"
          placeholder="Entity ID (e.g. cam-0_42)"
          value={entityId}
          onChange={(e) => setEntityId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && investigate()}
        />
        <button onClick={investigate} disabled={loading}
          className="bg-[var(--accent)] text-black px-4 py-2 rounded text-sm font-medium disabled:opacity-50">
          {loading ? "Investigating..." : "Investigate"}
        </button>
      </div>

      {report && (
        <div className="space-y-4">
          {/* Risk Assessment */}
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-2">Risk Assessment</h3>
            <p className={`text-sm font-mono ${riskColor}`}>{report.risk_assessment}</p>
          </div>

          {/* Narrative */}
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-2">Investigation Narrative</h3>
            <p className="text-sm text-[var(--text-muted)] whitespace-pre-wrap">{report.narrative}</p>
          </div>

          {/* Camera Route */}
          {report.camera_route.length > 0 && (
            <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Camera Route</h3>
              <div className="flex items-center gap-2 flex-wrap">
                {report.camera_route.map((r, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <span className="bg-[var(--accent)] text-black text-xs px-2 py-1 rounded">{r.camera_id}</span>
                    {i < report.camera_route.length - 1 && (
                      <span className="text-[var(--text-muted)] text-xs">
                        →{r.transit_time_s ? ` ${r.transit_time_s}s` : ""}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Timeline */}
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-2">Timeline ({report.timeline.length} events)</h3>
            <div className="space-y-1 max-h-60 overflow-y-auto">
              {report.timeline.slice(0, 30).map((evt, i) => (
                <div key={i} className="text-xs flex gap-3 border-b border-[var(--border)] py-1">
                  <span className="text-[var(--text-muted)] w-16 shrink-0">{evt.camera || "—"}</span>
                  <span className="font-mono w-24 shrink-0">{evt.type}</span>
                  <span className="text-[var(--text-muted)]">{evt.description}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Associates */}
          {report.associates.length > 0 && (
            <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Associates ({report.associates.length})</h3>
              <div className="space-y-2">
                {report.associates.map((a, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <div>
                      <span className="text-white font-mono">{a.entity_id}</span>
                      <span className="text-[var(--text-muted)] ml-2">{a.relationship}</span>
                      <span className="text-[var(--text-muted)] ml-2">({a.co_occurrences}x)</span>
                    </div>
                    <span className={a.risk_score > 0.5 ? "text-red-400" : "text-[var(--text-muted)]"}>
                      Risk: {(a.risk_score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommended Actions */}
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-2">Recommended Actions</h3>
            <ul className="space-y-1">
              {report.recommended_actions.map((a, i) => (
                <li key={i} className="text-sm text-[var(--text-muted)]">• {a}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
