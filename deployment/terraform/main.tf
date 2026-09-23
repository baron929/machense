terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "Google Cloud project used for optional model/data storage."
  type        = string
}

variable "region" {
  description = "Google Cloud region for the storage resource."
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Globally unique bucket name for non-sensitive model/data artifacts."
  type        = string
}

resource "google_storage_bucket" "model_artifacts" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

output "model_artifact_bucket" {
  description = "Bucket used for versioned model artifacts."
  value       = google_storage_bucket.model_artifacts.name
}
module "gcs_bucket" {
  source     = "./modules/gcs"
  bucket_name = "pm-mlops-data"
}

module "vm_instance" {
  source     = "./modules/compute"
  project_id = var.project_id
  region     = "us-central1"
}