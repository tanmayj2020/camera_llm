/**
 * TanStack Query hooks for VisionBrain API.
 *
 * Replaces manual useState + useEffect + setInterval patterns with
 * proper server state management: caching, deduplication, background refetch.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, fetchAPI } from "./api";

// ---- Cameras ----

export function useCameras() {
  return useQuery({ queryKey: ["cameras"], queryFn: api.listCameras, staleTime: 30_000 });
}

export function useAddCamera() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.addCamera,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cameras"] }),
  });
}

export function useDeleteCamera() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteCamera(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cameras"] }),
  });
}

// ---- Spatial & Graph ----

export function useSpatialState(cameraId: string, refetchMs = 2000) {
  return useQuery({
    queryKey: ["spatial", cameraId],
    queryFn: () => api.getSpatialState(cameraId),
    refetchInterval: refetchMs,
    enabled: !!cameraId,
  });
}

export function useRecentGraph(cameraId: string) {
  return useQuery({
    queryKey: ["graph", cameraId],
    queryFn: () => api.getRecentGraph(cameraId),
    refetchInterval: 5000,
    enabled: !!cameraId,
  });
}

// ---- KPIs ----

export function useKPIs(cameraId: string) {
  return useQuery({
    queryKey: ["kpi", cameraId],
    queryFn: () => api.getKPIs(cameraId),
    staleTime: 60_000,
    enabled: !!cameraId,
  });
}

export function useKPISummary(cameraId: string) {
  return useQuery({
    queryKey: ["kpi-summary", cameraId],
    queryFn: () => api.getKPISummary(cameraId),
    staleTime: 60_000,
    enabled: !!cameraId,
  });
}

// ---- Occupancy ----

export function useSiteOccupancy(refetchMs = 5000) {
  return useQuery({
    queryKey: ["occupancy-site"],
    queryFn: api.getSiteOccupancy,
    refetchInterval: refetchMs,
  });
}

export function useOccupancy(cameraId: string, refetchMs = 5000) {
  return useQuery({
    queryKey: ["occupancy", cameraId],
    queryFn: () => api.getOccupancy(cameraId),
    refetchInterval: refetchMs,
    enabled: !!cameraId,
  });
}

// ---- Floor Plan ----

export function useFloorPlan(refetchMs = 3000) {
  return useQuery({
    queryKey: ["floorplan"],
    queryFn: api.getFloorPlan,
    refetchInterval: refetchMs,
  });
}

// ---- Trackers ----

export function useTrackers() {
  return useQuery({ queryKey: ["trackers"], queryFn: api.listTrackers });
}

export function useCreateTracker() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (desc: string) => api.createTracker(desc),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["trackers"] }),
  });
}

export function useDeleteTracker() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteTracker(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["trackers"] }),
  });
}

// ---- Investigation ----

export function useInvestigate() {
  return useMutation({
    mutationFn: ({ entityId, sinceHours = 48 }: { entityId: string; sinceHours?: number }) =>
      api.investigate(entityId, sinceHours),
  });
}

// ---- Query / Chat ----

export function useQuery_VB() {
  return useMutation({
    mutationFn: ({ query, cameraId }: { query: string; cameraId?: string }) =>
      api.query(query, cameraId),
  });
}

// ---- Webhooks ----

export function useWebhooks() {
  return useQuery({ queryKey: ["webhooks"], queryFn: api.listWebhooks });
}

// ---- Fleet ----

export function useFleetStatus(refetchMs = 5000) {
  return useQuery({
    queryKey: ["fleet"],
    queryFn: () => fetchAPI("/api/fleet/status"),
    refetchInterval: refetchMs,
  });
}

// ---- Search ----

export function useForensicSearch() {
  return useMutation({
    mutationFn: ({ query, accessLevel, cameraId }: { query: string; accessLevel?: string; cameraId?: string }) =>
      api.forensicSearch(query, accessLevel, cameraId),
  });
}

// ---- Alerts ----

export function useAcknowledgeAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (eventId: string) => api.acknowledgeAlert(eventId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
}

// ---- System Health ----

export function useSystemHealth() {
  return useQuery({
    queryKey: ["system-health"],
    queryFn: () => fetchAPI("/api/system/health"),
    staleTime: 60_000,
  });
}

// ---- Natural Language Rule Compiler ----

export function useNLRules() {
  return useQuery({ queryKey: ["nl-rules"], queryFn: () => fetchAPI("/api/nl-rules") });
}

export function useCompileNLRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (instruction: string) =>
      fetchAPI("/api/nl-rules/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["nl-rules"] }),
  });
}

export function useDeleteNLRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ruleId: string) =>
      fetchAPI(`/api/nl-rules/${ruleId}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["nl-rules"] }),
  });
}

export function useToggleNLRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ ruleId, active }: { ruleId: string; active: boolean }) =>
      fetchAPI(`/api/nl-rules/${ruleId}/toggle`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["nl-rules"] }),
  });
}

// ---- Ambient Intelligence Score ----

export function useAmbientScore(zoneId: string, refetchMs = 3000) {
  return useQuery({
    queryKey: ["ambient", zoneId],
    queryFn: () => fetchAPI(`/api/ambient/${zoneId}`),
    refetchInterval: refetchMs,
    enabled: !!zoneId,
  });
}

export function useAllAmbientScores(refetchMs = 5000) {
  return useQuery({
    queryKey: ["ambient-all"],
    queryFn: () => fetchAPI("/api/ambient"),
    refetchInterval: refetchMs,
  });
}

export function useAmbientTrend(zoneId: string, minutes = 30) {
  return useQuery({
    queryKey: ["ambient-trend", zoneId, minutes],
    queryFn: () => fetchAPI(`/api/ambient/${zoneId}/trend?minutes=${minutes}`),
    enabled: !!zoneId,
  });
}

// ---- Predictive Path Interception ----

export function useActiveInterceptions(refetchMs = 2000) {
  return useQuery({
    queryKey: ["interceptions"],
    queryFn: () => fetchAPI("/api/interception/active"),
    refetchInterval: refetchMs,
  });
}

export function usePredictedTrajectories(refetchMs = 2000) {
  return useQuery({
    queryKey: ["trajectories"],
    queryFn: () => fetchAPI("/api/interception/trajectories"),
    refetchInterval: refetchMs,
  });
}

// ---- Gait DNA ----

export function useGaitGallery() {
  return useQuery({ queryKey: ["gait-gallery"], queryFn: () => fetchAPI("/api/gait/gallery") });
}

export function useGaitMatch(entityId: string) {
  return useQuery({
    queryKey: ["gait-match", entityId],
    queryFn: () => fetchAPI(`/api/gait/${entityId}/match`),
    enabled: !!entityId,
  });
}

// ---- Anomaly Contagion Network ----

export function useContagionGraph() {
  return useQuery({
    queryKey: ["contagion-graph"],
    queryFn: () => fetchAPI("/api/contagion/graph"),
    staleTime: 30_000,
  });
}

export function useContagionPrediction(zoneId: string) {
  return useQuery({
    queryKey: ["contagion-predict", zoneId],
    queryFn: () => fetchAPI(`/api/contagion/${zoneId}/predict`),
    enabled: !!zoneId,
  });
}

// ---- Scene Déjà Vu ----

export function useDejaVu(cameraId: string) {
  return useQuery({
    queryKey: ["deja-vu", cameraId],
    queryFn: () => fetchAPI(`/api/deja-vu/${cameraId}`),
    enabled: !!cameraId,
    staleTime: 10_000,
  });
}

export function useDejaVuStats() {
  return useQuery({
    queryKey: ["deja-vu-stats"],
    queryFn: () => fetchAPI("/api/deja-vu/stats"),
    staleTime: 60_000,
  });
}
