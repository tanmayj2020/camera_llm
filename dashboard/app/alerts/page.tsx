"use client";

import { useEffect, useState } from "react";
import { createAlertWebSocket } from "@/lib/api";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";

interface AlertItem {
  event_id: string;
  severity: string;
  anomaly_type: string;
  description: string;
  camera_id: string;
  causal?: { description: string; causal_explanation: string; recommended_action: string; grounded: boolean };
  acknowledged: boolean;
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);

  useEffect(() => {
    const ws = createAlertWebSocket((data: any) => {
      if (data.type === "alert" && data.anomaly) {
        setAlerts((prev) => [{
          ...data.anomaly,
          causal: data.causal,
          acknowledged: false,
        }, ...prev].slice(0, 100));
      }
    });
    return () => ws.close();
  }, []);

  const ackMutation = useMutation({
    mutationFn: (eventId: string) => api.acknowledgeAlert(eventId),
    onSuccess: (_, eventId) => {
      setAlerts((prev) => prev.map((a) => a.event_id === eventId ? { ...a, acknowledged: true } : a));
    },
  });

  const ack = (eventId: string) => ackMutation.mutate(eventId);

  const severityColor: Record<string, string> = {
    critical: "border-red-500 bg-red-950",
    high: "border-orange-500 bg-orange-950",
    medium: "border-yellow-500 bg-yellow-950",
    low: "border-gray-500 bg-gray-900",
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Alert Feed</h2>
      <div className="space-y-4">
        {alerts.length === 0 && <p className="text-sm text-[var(--text-muted)]">No alerts. System monitoring…</p>}
        {alerts.map((alert, i) => (
          <div key={i} className={`p-4 rounded-lg border ${severityColor[alert.severity] || severityColor.low}`}>
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-bold uppercase">{alert.severity}</span>
                  <span className="text-xs text-[var(--text-muted)]">{alert.anomaly_type}</span>
                  <span className="text-xs text-[var(--text-muted)]">{alert.camera_id}</span>
                </div>
                <p className="text-sm mb-2">{alert.description}</p>
                {alert.causal && (
                  <div className="text-xs text-[var(--text-muted)] space-y-1">
                    <p><strong>Cause:</strong> {alert.causal.causal_explanation}</p>
                    <p><strong>Action:</strong> {alert.causal.recommended_action}</p>
                    {alert.causal.grounded && <span className="text-green-400">✓ Evidence verified</span>}
                  </div>
                )}
              </div>
              {!alert.acknowledged && (
                <button onClick={() => ack(alert.event_id)}
                  className="text-xs bg-white/10 hover:bg-white/20 px-3 py-1 rounded">
                  Acknowledge
                </button>
              )}
              {alert.acknowledged && <span className="text-xs text-green-400">✓ Acked</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
