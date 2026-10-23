# variables: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# Physical AI Robotics - Terraform Variables
# AWS Infrastructure Configuration
# ============================================================

# Note:
# Variables already declared in compute.tf, database.tf,
# and gpu.tf must not be declared again here.

# Networking
variable "vpc_cidr" {
  description = "CIDR block for the project VPC"
  type        = string
  default     = "10.20.0.0/16"
}

# Application Configuration
variable "app_port" {
  description = "FastAPI backend port"
  type        = number
  default     = 8000
}

variable "ml_port" {
  description = "Machine learning inference port"
  type        = number
  default     = 8001
}

variable "simulation_only" {
  description = "Restrict application to simulation mode"
  type        = bool
  default     = true

  validation {
    condition     = var.simulation_only
    error_message = "This deployment must remain simulation-only."
  }
}

# Storage
variable "storage_retention_days" {
  description = "Default log retention in days"
  type        = number
  default     = 30
}

# Monitoring
variable "cpu_alarm_threshold" {
  description = "CloudWatch CPU alarm threshold"
  type        = number
  default     = 80
}

# Resource Tags
variable "owner" {
  description = "Infrastructure owner"
  type        = string
  default     = "physical-ai-team"
}

# Deployment Controls
variable "enable_monitoring" {
  type    = bool
  default = true
}

variable "enable_backups" {
  type    = bool
  default = true
}

variable "enable_gpu_monitoring" {
  type    = bool
  default = false
}
