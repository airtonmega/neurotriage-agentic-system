# ============================================================================
# Terraform Configuration for NeuroTriage-AI on GCP
# ============================================================================
# This configuration deploys:
# - Google Pub/Sub topics and subscriptions
# - Cloud Run services for processing
# - Secret Manager for credentials
# - IAM bindings for least-privilege access
# ============================================================================

terraform {
  required_version = ">= 1.9.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.60"
    }
  }
  
  backend "gcs" {
    bucket = "neurotriage-terraform-state"
    prefix = "terraform/state"
  }
}

# ============================================================================
# VARIABLES
# ============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "databricks_host" {
  description = "Databricks workspace URL"
  type        = string
  sensitive   = true
}

variable "databricks_token" {
  description = "Databricks access token"
  type        = string
  sensitive   = true
}

# ============================================================================
# PROVIDERS
# ============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# ============================================================================
# LOCALS
# ============================================================================

locals {
  service_name = "neurotriage-ai"
  labels = {
    project     = "neurotriage-ai"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ============================================================================
# ENABLE APIS
# ============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ============================================================================
# SERVICE ACCOUNT
# ============================================================================

resource "google_service_account" "neurotriage" {
  account_id   = "neurotriage-sa"
  display_name = "NeuroTriage AI Service Account"
  description  = "Service account for NeuroTriage AI processing"
}

# IAM bindings for the service account
resource "google_project_iam_member" "neurotriage_roles" {
  for_each = toset([
    "roles/pubsub.subscriber",
    "roles/pubsub.publisher",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.neurotriage.email}"
}

# ============================================================================
# PUB/SUB TOPICS & SUBSCRIPTIONS
# ============================================================================

# Audio input topic
resource "google_pubsub_topic" "audio_input" {
  name   = "${local.service_name}-audio-input"
  labels = local.labels
  
  message_retention_duration = "86400s"  # 24 hours
  
  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Audio input subscription
resource "google_pubsub_subscription" "audio_input" {
  name  = "${local.service_name}-audio-sub"
  topic = google_pubsub_topic.audio_input.name
  
  labels = local.labels
  
  ack_deadline_seconds       = 60
  message_retention_duration = "604800s"  # 7 days
  retain_acked_messages      = true
  
  expiration_policy {
    ttl = ""  # Never expire
  }
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
  
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }
}

# Results topic
resource "google_pubsub_topic" "results" {
  name   = "${local.service_name}-results"
  labels = local.labels
  
  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Dead letter topic
resource "google_pubsub_topic" "dead_letter" {
  name   = "${local.service_name}-dead-letter"
  labels = local.labels
  
  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Emergency alerts topic (high priority)
resource "google_pubsub_topic" "emergency_alerts" {
  name   = "${local.service_name}-emergency-alerts"
  labels = local.labels
  
  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# ============================================================================
# SECRET MANAGER
# ============================================================================

resource "google_secret_manager_secret" "databricks_token" {
  secret_id = "${local.service_name}-databricks-token"
  
  labels = local.labels
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "databricks_token" {
  secret      = google_secret_manager_secret.databricks_token.id
  secret_data = var.databricks_token
}

resource "google_secret_manager_secret" "pii_salt" {
  secret_id = "${local.service_name}-pii-salt"
  
  labels = local.labels
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

# ============================================================================
# ARTIFACT REGISTRY
# ============================================================================

resource "google_artifact_registry_repository" "neurotriage" {
  location      = var.region
  repository_id = local.service_name
  description   = "Container images for NeuroTriage AI"
  format        = "DOCKER"
  
  labels = local.labels
  
  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}

# ============================================================================
# CLOUD RUN SERVICE
# ============================================================================

resource "google_cloud_run_v2_service" "processor" {
  name     = "${local.service_name}-processor"
  location = var.region
  
  labels = local.labels
  
  template {
    service_account = google_service_account.neurotriage.email
    
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
    
    timeout = "300s"
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${local.service_name}/processor:latest"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle = true
      }
      
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      
      env {
        name  = "PUBSUB_SUBSCRIPTION"
        value = google_pubsub_subscription.audio_input.name
      }
      
      env {
        name = "DATABRICKS_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.databricks_token.secret_id
            version = "latest"
          }
        }
      }
      
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 3
        failure_threshold     = 3
      }
      
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }
  
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  
  depends_on = [
    google_project_service.apis["run.googleapis.com"],
    google_artifact_registry_repository.neurotriage,
  ]
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "service_url" {
  description = "URL of the Cloud Run processor service"
  value       = google_cloud_run_v2_service.processor.uri
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value = {
    audio_input      = google_pubsub_topic.audio_input.name
    results          = google_pubsub_topic.results.name
    emergency_alerts = google_pubsub_topic.emergency_alerts.name
    dead_letter      = google_pubsub_topic.dead_letter.name
  }
}

output "service_account_email" {
  description = "Service account email for the processor"
  value       = google_service_account.neurotriage.email
}

output "artifact_registry_url" {
  description = "Artifact Registry URL for container images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${local.service_name}"
}
