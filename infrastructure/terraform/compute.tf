# compute: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# PHYSICAL AI ROBOTICS - AWS COMPUTE
# File: terraform/compute.tf
# ============================================================

variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "project_name" {
  type    = string
  default = "physical-ai"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_id" {
  type = string
}

variable "ami_id" {
  type        = string
  description = "Compatible Linux AMI ID"
}

variable "instance_profile_name" {
  type        = string
  description = "Existing IAM instance profile"
}

variable "backend_instance_type" {
  type    = string
  default = "t3.medium"
}

variable "ml_instance_type" {
  type    = string
  default = "t3.large"
}

variable "worker_instance_type" {
  type    = string
  default = "t3.medium"
}

# ------------------------------------------------------------
# SECURITY GROUP
# ------------------------------------------------------------

resource "aws_security_group" "compute" {
  name_prefix = "${var.project_name}-compute-"
  description = "Private compute instances"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-compute-sg"
    Environment = var.environment
  }
}

# ------------------------------------------------------------
# COMMON INSTANCE SETTINGS
# ------------------------------------------------------------

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    RobotMode   = "simulation"
  }
}

# ------------------------------------------------------------
# FASTAPI BACKEND
# ------------------------------------------------------------

resource "aws_instance" "backend" {
  ami                    = var.ami_id
  instance_type          = var.backend_instance_type
  subnet_id              = var.private_subnet_id
  vpc_security_group_ids = [aws_security_group.compute.id]
  iam_instance_profile   = var.instance_profile_name

  associate_public_ip_address = false

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-backend"
    Role = "fastapi"
  })
}

# ------------------------------------------------------------
# ML INFERENCE SERVER
# ------------------------------------------------------------

resource "aws_instance" "ml" {
  ami                    = var.ami_id
  instance_type          = var.ml_instance_type
  subnet_id              = var.private_subnet_id
  vpc_security_group_ids = [aws_security_group.compute.id]
  iam_instance_profile   = var.instance_profile_name

  associate_public_ip_address = false

  root_block_device {
    volume_size = 80
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-ml"
    Role = "ml-inference"
  })
}

# ------------------------------------------------------------
# CELERY BACKGROUND WORKER
# ------------------------------------------------------------

resource "aws_instance" "worker" {
  ami                    = var.ami_id
  instance_type          = var.worker_instance_type
  subnet_id              = var.private_subnet_id
  vpc_security_group_ids = [aws_security_group.compute.id]
  iam_instance_profile   = var.instance_profile_name

  associate_public_ip_address = false

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-worker"
    Role = "celery-worker"
  })
}

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

output "backend_private_ip" {
  value = aws_instance.backend.private_ip
}

output "ml_private_ip" {
  value = aws_instance.ml.private_ip
}

output "worker_private_ip" {
  value = aws_instance.worker.private_ip
}

output "compute_security_group_id" {
  value = aws_security_group.compute.id
}
