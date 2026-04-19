import "./globals.css";
import type { Metadata } from "next";
import QueryProvider from "@/components/query-provider";

export const metadata: Metadata = {
  title: "VisionBrain — Intelligent CCTV Analytics",
  description: "AI-powered CCTV analytics with real-time reasoning and action",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[var(--bg)]">
        <QueryProvider>
          <nav className="border-b border-[var(--border)] px-6 py-3 flex flex-wrap items-center gap-4">
            <a href="/" className="text-lg font-bold text-[var(--accent)] mr-2">🧠 VisionBrain</a>
            <a href="/cameras" className="text-sm text-[var(--text-muted)] hover:text-white">Cameras</a>
            <a href="/videowall" className="text-sm text-[var(--text-muted)] hover:text-white">Video Wall</a>
            <a href="/alerts" className="text-sm text-[var(--text-muted)] hover:text-white">Alerts</a>
            <a href="/chat" className="text-sm text-[var(--text-muted)] hover:text-white">Chat</a>
            <a href="/kpi" className="text-sm text-[var(--text-muted)] hover:text-white">KPIs</a>
            <a href="/occupancy" className="text-sm text-[var(--text-muted)] hover:text-white">Occupancy</a>
            <a href="/graph" className="text-sm text-[var(--text-muted)] hover:text-white">Graph</a>
            <a href="/trackers" className="text-sm text-[var(--text-muted)] hover:text-white">Trackers</a>
            <a href="/investigate" className="text-sm text-[var(--text-muted)] hover:text-white">Investigate</a>
            <a href="/floorplan" className="text-sm text-[var(--text-muted)] hover:text-white">Floor Plan</a>
            <a href="/zones" className="text-sm text-[var(--text-muted)] hover:text-white">Zones</a>
            <a href="/timeline" className="text-sm text-[var(--text-muted)] hover:text-white">Timeline</a>
            <a href="/search" className="text-sm text-[var(--text-muted)] hover:text-white">Search</a>
            <a href="/webhooks" className="text-sm text-[var(--text-muted)] hover:text-white">Webhooks</a>
            <a href="/fleet" className="text-sm text-[var(--text-muted)] hover:text-white">Fleet</a>
            <a href="/journey" className="text-sm text-[var(--text-muted)] hover:text-white">Journey</a>
          </nav>
          <main className="p-6">{children}</main>
        </QueryProvider>
      </body>
    </html>
  );
}
