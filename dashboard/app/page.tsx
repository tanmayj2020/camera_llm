"use client";

import { useEffect, useState } from "react";
import { createAlertWebSocket } from "@/lib/api";

interface Alert {
  type: string;
  anomaly?: { severity: string; anomaly_type: string; description: string; camera_id: string };
  prediction?: { prediction_type: string; description: string; confidence: number };
}

export default function HomePage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [cameras] = useState(["cam-0", "cam-1", "cam-2", "cam-3"]);

  useEffect(() => {
    const ws = createAlertWebSocket((data: Alert) => {
      setAlerts((prev) => [data, ...prev].slice(0, 50));
    });
    return () => ws.close();
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Camera Grid */}
      <div className="lg:col-span-2">
        <h2 className="text-lg font-semibold mb-4">Live Cameras</h2>
        <div className="grid grid-cols-2 gap-4">
          {cameras.map((cam) => (
            <div key={cam} className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 aspect-video flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl mb-2">📹</div>
                <p className="text-sm text-[var(--text-muted)]">{cam}</p>
                <span className="inline-block mt-1 px-2 py-0.5 text-xs rounded bg-green-900 text-green-300">Live</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Alert Feed */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Real-Time Alerts</h2>
        <div className="space-y-3 max-h-[600px] overflow-y-auto">
          {alerts.length === 0 && (
            <p className="text-sm text-[var(--text-muted)]">No alerts yet. Monitoring…</p>
          )}
          {alerts.map((alert, i) => (
            <div key={i} className={`p-3 rounded-lg border ${
              alert.type === "alert"
                ? alert.anomaly?.severity === "critical" ? "border-red-500 bg-red-950"
                : alert.anomaly?.severity === "high" ? "border-orange-500 bg-orange-950"
                : "border-yellow-500 bg-yellow-950"
                : "border-blue-500 bg-blue-950"
            }`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono">
                  {alert.type === "alert" ? "🚨" : "🔮"}
                </span>
                <span className="text-xs font-semibold uppercase">
                  {alert.type === "alert" ? alert.anomaly?.severity : "prediction"}
                </span>
              </div>
              <p className="text-sm">
                {alert.type === "alert"
                  ? alert.anomaly?.description
                  : alert.prediction?.description}
              </p>
              {alert.anomaly?.camera_id && (
                <p className="text-xs text-[var(--text-muted)] mt-1">{alert.anomaly.camera_id}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
