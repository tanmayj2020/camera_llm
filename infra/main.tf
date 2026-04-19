variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  default = "us-central1"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Pub/Sub ---
resource "google_pubsub_topic" "events" {
  name = "visionbrain-events"
}

resource "google_pubsub_subscription" "events_sub" {
  name  = "visionbrain-events-sub"
  topic = google_pubsub_topic.events.id
  ack_deadline_seconds = 30

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.events_dlq.id
    max_delivery_attempts = 5
  }
}

resource "google_pubsub_topic" "events_dlq" {
  name = "visionbrain-events-dlq"
}

# --- Cloud Storage ---
resource "google_storage_bucket" "keyframes" {
  name     = "${var.project_id}-visionbrain-keyframes"
  location = var.region

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }
  }
}

# --- BigQuery ---
resource "google_bigquery_dataset" "visionbrain" {
  dataset_id = "visionbrain"
  location   = var.region
}

resource "google_bigquery_table" "events" {
  dataset_id = google_bigquery_dataset.visionbrain.dataset_id
  table_id   = "events"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  schema = jsonencode([
    { name = "event_id",       type = "STRING",    mode = "REQUIRED" },
    { name = "timestamp",      type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "camera_id",      type = "STRING",    mode = "REQUIRED" },
    { name = "event_type",     type = "STRING" },
    { name = "scene_activity", type = "FLOAT" },
    { name = "objects",        type = "JSON" },
    { name = "audio_events",   type = "JSON" },
    { name = "keyframe_gcs",   type = "STRING" },
  ])
}

# --- GKE Autopilot ---
resource "google_container_cluster" "main" {
  name     = "visionbrain-cluster"
  location = var.region

  enable_autopilot = true

  release_channel {
    channel = "REGULAR"
  }
}

# --- VPC ---
resource "google_compute_network" "vpc" {
  name                    = "visionbrain-vpc"
  auto_create_subnetworks = true
}

output "pubsub_topic" {
  value = google_pubsub_topic.events.name
}

output "gke_cluster" {
  value = google_container_cluster.main.name
}

output "keyframe_bucket" {
  value = google_storage_bucket.keyframes.name
}
