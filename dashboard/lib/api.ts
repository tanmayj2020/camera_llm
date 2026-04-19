const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchAPI(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function createAlertWebSocket(onMessage: (data: any) => void): WebSocket {
  const wsUrl = API_BASE.replace("http", "ws") + "/ws/alerts";
  const ws = new WebSocket(wsUrl);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}

export const api = {
  // Events
  ingestEvent: (event: any) => fetchAPI("/api/events", { method: "POST", body: JSON.stringify(event) }),

  // Query & Chat
  query: (query: string, cameraId?: string) =>
    fetchAPI("/api/query", { method: "POST", body: JSON.stringify({ query, camera_id: cameraId }) }),
  converse: (message: string, sessionId?: string, cameraId?: string) =>
    fetchAPI("/api/converse", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId, camera_id: cameraId }),
    }),

  // KPIs
  getKPIs: (cameraId: string) => fetchAPI(`/api/kpi/${cameraId}`),
  getKPISummary: (cameraId: string) => fetchAPI(`/api/kpi/${cameraId}/summary`),

  // Spatial & Graph
  getSpatialState: (cameraId: string) => fetchAPI(`/api/spatial/${cameraId}`),
  getRecentGraph: (cameraId: string) => fetchAPI(`/api/graph/${cameraId}/recent`),

  // Alerts
  acknowledgeAlert: (eventId: string) => fetchAPI(`/api/alerts/${eventId}/ack`, { method: "POST" }),
  addRule: (rule: any) => fetchAPI("/api/rules", { method: "POST", body: JSON.stringify(rule) }),

  // Custom trackers
  listTrackers: () => fetchAPI("/api/trackers"),
  createTracker: (description: string) =>
    fetchAPI("/api/trackers", { method: "POST", body: JSON.stringify({ description }) }),
  updateTracker: (trackerId: string, status: string) =>
    fetchAPI(`/api/trackers/${trackerId}`, { method: "PATCH", body: JSON.stringify({ status }) }),
  deleteTracker: (trackerId: string) =>
    fetchAPI(`/api/trackers/${trackerId}`, { method: "DELETE" }),

  // Story & profiles
  getStory: (entityId: string, sinceHours = 24) =>
    fetchAPI(`/api/story/${entityId}?since_hours=${sinceHours}`),
  getProfile: (entityId: string) => fetchAPI(`/api/profile/${entityId}`),

  // Investigation
  investigate: (entityId: string, sinceHours = 48) =>
    fetchAPI(`/api/investigate/${entityId}?since_hours=${sinceHours}`, { method: "POST" }),

  // Video synopsis
  getSynopsis: (cameraId: string, hours = 4) =>
    fetchAPI(`/api/synopsis/${cameraId}?hours=${hours}`),

  // Floor plan
  getFloorPlan: () => fetchAPI("/api/floorplan"),

  // Forensic search
  forensicSearch: (query: string, accessLevel = "anonymous", cameraId?: string) =>
    fetchAPI("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, access_level: accessLevel, camera_id: cameraId }),
    }),
  unlockResult: (resultId: string, userId: string, accessLevel: string, justification: string) =>
    fetchAPI("/api/search/unlock", {
      method: "POST",
      body: JSON.stringify({ result_id: resultId, user_id: userId, access_level: accessLevel, justification }),
    }),
  getSearchAudit: () => fetchAPI("/api/search/audit"),

  // Cameras
  listCameras: () => fetchAPI("/api/cameras"),
  addCamera: (camera: any) => fetchAPI("/api/cameras", { method: "POST", body: JSON.stringify(camera) }),
  updateCamera: (id: string, data: any) =>
    fetchAPI(`/api/cameras/${id}`, { method: "PATCH", body: JSON.stringify(data) }),
  deleteCamera: (id: string) => fetchAPI(`/api/cameras/${id}`, { method: "DELETE" }),

  // Webhooks
  listWebhooks: () => fetchAPI("/api/webhooks"),
  createWebhook: (webhook: any) =>
    fetchAPI("/api/webhooks", { method: "POST", body: JSON.stringify(webhook) }),
  updateWebhook: (id: string, data: any) =>
    fetchAPI(`/api/webhooks/${id}`, { method: "PATCH", body: JSON.stringify(data) }),
  deleteWebhook: (id: string) => fetchAPI(`/api/webhooks/${id}`, { method: "DELETE" }),
  getWebhookLogs: (id: string) => fetchAPI(`/api/webhooks/${id}/logs`),

  // Occupancy
  getOccupancy: (cameraId: string) => fetchAPI(`/api/occupancy/${cameraId}`),
  getSiteOccupancy: () => fetchAPI("/api/occupancy"),

  // Compliance
  getCompliance: (cameraId: string) => fetchAPI(`/api/compliance/${cameraId}`),
  addSOPRule: (rule: any) =>
    fetchAPI("/api/compliance/sop", { method: "POST", body: JSON.stringify(rule) }),

  // LPR
  getLPRLog: (cameraId: string) => fetchAPI(`/api/lpr/log/${cameraId}`),
  addToWatchlist: (plate: string, reason: string, severity = "high") =>
    fetchAPI("/api/lpr/watchlist", { method: "POST", body: JSON.stringify({ plate, reason, severity }) }),
  getWatchlist: () => fetchAPI("/api/lpr/watchlist"),
  removeFromWatchlist: (plate: string) =>
    fetchAPI(`/api/lpr/watchlist/${plate}`, { method: "DELETE" }),

  // Reports
  listReportSchedules: () => fetchAPI("/api/reports/schedules"),
  createReportSchedule: (schedule: any) =>
    fetchAPI("/api/reports/schedules", { method: "POST", body: JSON.stringify(schedule) }),
  deleteReportSchedule: (id: string) =>
    fetchAPI(`/api/reports/schedules/${id}`, { method: "DELETE" }),
  generateReport: (id: string) =>
    fetchAPI(`/api/reports/generate/${id}`, { method: "POST" }),

  // Digest
  getDigest: (hours = 24, cameraId?: string) =>
    fetchAPI(`/api/digest?hours=${hours}${cameraId ? `&camera_id=${cameraId}` : ""}`),

  // Zones
  addZone: (zone: any) => fetchAPI("/api/zones", { method: "POST", body: JSON.stringify(zone) }),
  deleteZone: (zoneId: string) => fetchAPI(`/api/zones/${zoneId}`, { method: "DELETE" }),

  // Timeline (alias for getRecentGraph)
  getTimeline: (cameraId: string) => fetchAPI(`/api/graph/${cameraId}/recent`),

  // Auth / API Keys
  createApiKey: (data: any) =>
    fetchAPI("/api/auth/keys", { method: "POST", body: JSON.stringify(data) }),
  listApiKeys: () => fetchAPI("/api/auth/keys"),
  deleteApiKey: (keyHash: string) =>
    fetchAPI(`/api/auth/keys/${keyHash}`, { method: "DELETE" }),

  // Crowd Analytics
  getCrowd: (cameraId: string) => fetchAPI(`/api/crowd/${cameraId}`),

  // Scene Graph
  getSceneGraph: (cameraId: string) => fetchAPI(`/api/scene-graph/${cameraId}`),

  // Contextual Normality
  getNormality: (cameraId: string) => fetchAPI(`/api/normality/${cameraId}`),

  // Counterfactual Explanation
  getCounterfactual: (eventId: string) =>
    fetchAPI(`/api/explain/counterfactual/${eventId}`, { method: "POST" }),

  // Collective Anomaly
  getCollectiveAnomaly: (cameraId: string) => fetchAPI(`/api/anomaly/collective/${cameraId}`),

  // What-If Simulation
  simulate: (scenarioType: string, parameters: any = {}, description = "") =>
    fetchAPI("/api/simulate", {
      method: "POST",
      body: JSON.stringify({ scenario_type: scenarioType, parameters, description }),
    }),

  // System Health
  getSystemHealth: () => fetchAPI("/api/system/health"),

  // Dwell Analytics
  getDwellAnalytics: (zoneId: string) => fetchAPI(`/api/analytics/dwell/${zoneId}`),

  // Heatmap
  getHeatmap: (cameraId: string, type = "movement") =>
    fetchAPI(`/api/heatmap/${cameraId}?type=${type}`),

  // Fleet
  getFleetStatus: () => fetchAPI("/api/fleet/status"),
  sendHeartbeat: (data: any) =>
    fetchAPI("/api/fleet/heartbeat", { method: "POST", body: JSON.stringify(data) }),

  // Proactive Agent
  getProactivePriorities: () => fetchAPI("/api/proactive/priorities"),
  setProactivePriorities: (priorities: string[]) =>
    fetchAPI("/api/proactive/priorities", { method: "POST", body: JSON.stringify({ priorities }) }),
  addWatchEntity: (entityId: string) =>
    fetchAPI("/api/proactive/watch", { method: "POST", body: JSON.stringify({ entity_id: entityId }) }),

  // Consent & GDPR
  getConsent: (entityId: string) => fetchAPI(`/api/consent/${entityId}`),
  recordConsent: (data: any) =>
    fetchAPI("/api/consent", { method: "POST", body: JSON.stringify(data) }),
  getDPIAReport: () => fetchAPI("/api/consent/dpia-report"),
  enforceRetention: () => fetchAPI("/api/retention/enforce", { method: "POST" }),

  // Evidence Package
  buildEvidence: (incidentId: string, cameraIds: string[] = [], hours = 4) =>
    fetchAPI("/api/evidence/build", {
      method: "POST",
      body: JSON.stringify({ incident_id: incidentId, camera_ids: cameraIds, time_range_hours: hours }),
    }),

  // Shift Handover
  getShiftBriefing: (hours = 8) => fetchAPI(`/api/shift/briefing?hours=${hours}`),

  // Camera Programs
  createCameraProgram: (instruction: string) =>
    fetchAPI("/api/camera-programs", { method: "POST", body: JSON.stringify({ instruction }) }),
  listCameraPrograms: () => fetchAPI("/api/camera-programs"),
  deleteCameraProgram: (id: string) => fetchAPI(`/api/camera-programs/${id}`, { method: "DELETE" }),

  // Adversarial Robustness
  getAdversarialHealth: () => fetchAPI("/api/adversarial/health"),

  // Entity Journey
  getEntityJourney: (entityId: string) => fetchAPI(`/api/journey/${entityId}`),
  getActiveJourneys: () => fetchAPI("/api/journey/active"),
  getJourneyPath: (entityId: string) => fetchAPI(`/api/journey/${entityId}/path`),

  // Retail Analytics
  getRetailMetrics: (storeId = "default") => fetchAPI(`/api/retail/metrics?store_id=${storeId}`),
  configureRetail: (config: any) =>
    fetchAPI("/api/retail/configure", { method: "POST", body: JSON.stringify(config) }),
  getRetailFunnel: () => fetchAPI("/api/retail/funnel"),

  // Event Bus
  subscribeEvents: (name: string, eventTypes: string[], filters: any, callbackUrl: string) =>
    fetchAPI("/api/events/subscribe", {
      method: "POST",
      body: JSON.stringify({ subscriber_name: name, event_types: eventTypes, filters, callback_url: callbackUrl }),
    }),
  listEventSubscriptions: () => fetchAPI("/api/events/subscriptions"),
  unsubscribeEvents: (id: string) => fetchAPI(`/api/events/subscriptions/${id}`, { method: "DELETE" }),
  getEventLog: (eventType?: string) =>
    fetchAPI(`/api/events/log${eventType ? `?event_type=${eventType}` : ""}`),

  // Visual Grounding
  getGroundedEntities: (cameraId: string) => fetchAPI(`/api/grounding/${cameraId}`),

  // Staffing & Resources
  predictDemand: (cameraId = "all", hours = 24) =>
    fetchAPI(`/api/staffing/predict?camera_id=${cameraId}&hours=${hours}`),
  getStaffingRecommendation: () => fetchAPI("/api/staffing/recommend"),
  getResourcePlan: () => fetchAPI("/api/staffing/plan"),

  // Behavioral Stress
  getStressAssessments: () => fetchAPI("/api/behavioral/stress"),

  // Federation
  registerFederationPeer: (siteId: string, callbackUrl: string) =>
    fetchAPI("/api/federation/peers", {
      method: "POST", body: JSON.stringify({ site_id: siteId, callback_url: callbackUrl }),
    }),
  listFederationPeers: () => fetchAPI("/api/federation/peers"),
  shareFederationAlert: (entityId: string, severity: string, description: string) =>
    fetchAPI("/api/federation/share", {
      method: "POST", body: JSON.stringify({ entity_id: entityId, severity, description }),
    }),
  getFederationAlerts: (hours = 24) => fetchAPI(`/api/federation/alerts?since_hours=${hours}`),
  checkFederationEntity: (entityId: string) => fetchAPI(`/api/federation/check/${entityId}`),
  getFederationStatus: () => fetchAPI("/api/federation/status"),
};
