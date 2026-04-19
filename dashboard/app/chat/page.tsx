"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  evidence?: any[];
  confidence?: number;
  complexity?: string;
  latency_ms?: number;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [cameraId, setCameraId] = useState("cam-0");

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.query(input, cameraId);
      const assistantMsg: Message = {
        role: "assistant",
        content: res.answer,
        evidence: res.evidence,
        confidence: res.confidence,
        complexity: res.complexity,
        latency_ms: res.latency_ms,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error processing query." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Video Intelligence Chat</h2>

      <div className="mb-4 flex gap-2 items-center">
        <label className="text-sm text-[var(--text-muted)]">Camera:</label>
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-2 py-1 text-sm">
          {["cam-0", "cam-1", "cam-2", "cam-3"].map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4 min-h-[400px] max-h-[600px] overflow-y-auto mb-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-lg p-3 ${
              msg.role === "user" ? "bg-[var(--accent)] text-white" : "bg-[var(--border)]"
            }`}>
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              {msg.confidence !== undefined && (
                <div className="mt-2 flex gap-3 text-xs text-[var(--text-muted)]">
                  <span>Confidence: {(msg.confidence * 100).toFixed(0)}%</span>
                  <span>Complexity: {msg.complexity}</span>
                  <span>{msg.latency_ms?.toFixed(0)}ms</span>
                </div>
              )}
              {msg.evidence && msg.evidence.length > 0 && (
                <div className="mt-2 text-xs text-[var(--text-muted)]">
                  📎 {msg.evidence.length} evidence segments
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && <div className="text-sm text-[var(--text-muted)]">Thinking…</div>}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about your cameras…"
          className="flex-1 bg-[var(--surface)] border border-[var(--border)] rounded-lg px-4 py-2 text-sm"
        />
        <button onClick={sendMessage} disabled={loading}
          className="bg-[var(--accent)] text-white px-4 py-2 rounded-lg text-sm hover:opacity-90 disabled:opacity-50">
          Send
        </button>
      </div>
    </div>
  );
}
