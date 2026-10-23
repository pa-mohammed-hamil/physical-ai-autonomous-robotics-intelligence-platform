
# ============================================================
# PHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE PLATFORM
# File: terraform/main.tf
# AWS | Terraform | AI/ML Infrastructure
# ============================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

# ------------------------------------------------------------
# AWS PROVIDER
# ------------------------------------------------------------

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Application = "Physical-AI-Robotics"
    }
  }
}

provider "random" {}

# ------------------------------------------------------------
# AWS ACCOUNT INFORMATION
# ------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# ------------------------------------------------------------
# PROJECT IDENTIFIERS
# ------------------------------------------------------------

locals {
  resource_prefix = "${var.project_name}-${var.environment}"

  simulation_mode = true

  application_config = {
    APP_ENV                          = var.environment
    ROBOT_MODE                       = "simulation"
    SIMULATION_ONLY                  = "true"
    PHYSICAL_ROBOT_CONTROL_ENABLED   = "false"
    HARDWARE_ACTUATION_ENABLED       = "false"
    ACTUATOR_COMMANDS_ENABLED        = "false"
    REMOTE_MOTION_COMMANDS_ENABLED   = "false"
    EMERGENCY_STOP_RESET_ENABLED     = "false"
    SAFETY_INTERLOCKS_REQUIRED       = "true"
  }
}

# ------------------------------------------------------------
# AWS OUTPUTS
# ------------------------------------------------------------

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "deployed_region" {
  description = "AWS deployment region"
  value       = data.aws_region.current.name
}

output "resource_prefix" {
  description = "Resource naming prefix"
  value       = local.resource_prefix
}

output "robot_simulation_mode" {
  description = "Configured robotics operation mode"
  value       = "simulation"
}
