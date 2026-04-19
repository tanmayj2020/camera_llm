"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function KPIPage() {
  const [cameraId, setCameraId] = useState("cam-0");
  const [kpis, setKpis] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);

  useEffect(() => {
    api.getKPIs(cameraId).then(setKpis).catch(() => {});
    api.getKPISummary(cameraId).then(setSummary).catch(() => {});
  }, [cameraId]);

  const MetricCard = ({ label, value, trend }: { label: string; value: any; trend?: any }) => (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
      <p className="text-xs text-[var(--text-muted)] mb-1">{label}</p>
      <p className="text-2xl font-bold">{value ?? "—"}</p>
      {trend && (
        <p className={`text-xs mt-1 ${trend.direction === "up" ? "text-green-400" : trend.direction === "down" ? "text-red-400" : "text-gray-400"}`}>
          {trend.direction === "up" ? "↑" : trend.direction === "down" ? "↓" : "→"} {Math.abs(trend.change_pct)}%
        </p>
      )}
    </div>
  );

  return (
    <div>
      <div className="flex items-center gap-4 mb-6">
        <h2 className="text-lg font-semibold">Business KPIs</h2>
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-2 py-1 text-sm">
          {["cam-0", "cam-1", "cam-2", "cam-3"].map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Foot Traffic" value={kpis?.foot_traffic} trend={summary?.trends?.foot_traffic} />
        <MetricCard label="Peak Hour" value={kpis?.peak_hour ? `${kpis.peak_hour}:00` : "—"} />
        <MetricCard label="Incidents" value={kpis?.incident_count} trend={summary?.trends?.incident_count} />
        <MetricCard label="Audio Alerts" value={kpis?.audio_alerts} />
      </div>

      {summary?.narrative && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-6 mb-6">
          <h3 className="text-sm font-semibold mb-2">Daily Summary</h3>
          <p className="text-sm text-[var(--text-muted)] whitespace-pre-wrap">{summary.narrative}</p>
        </div>
      )}

      {summary?.recommendations && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-6">
          <h3 className="text-sm font-semibold mb-2">Recommendations</h3>
          <ul className="space-y-2">
            {summary.recommendations.map((rec: string, i: number) => (
              <li key={i} className="text-sm text-[var(--text-muted)] flex gap-2">
                <span>💡</span> {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
