# gpu: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# PHYSICAL AI ROBOTICS - GPU COMPUTE
# File: terraform/gpu.tf
# AWS EC2 | NVIDIA GPU | AI/ML Workloads
# ============================================================

variable "gpu_instance_type" {
  description = "AWS GPU instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_enabled" {
  description = "Enable GPU instance deployment"
  type        = bool
  default     = false
}

variable "gpu_volume_size" {
  description = "GPU instance root volume in GB"
  type        = number
  default     = 100
}

variable "gpu_ami_id" {
  description = "GPU-compatible Linux AMI with NVIDIA drivers"
  type        = string
}

# ------------------------------------------------------------
# GPU SECURITY GROUP
# ------------------------------------------------------------

resource "aws_security_group" "gpu" {
  name_prefix = "${var.project_name}-gpu-"
  description = "Private GPU compute security group"
  vpc_id      = var.vpc_id

  egress {
    description = "Outbound access for dependencies"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-gpu-sg"
  })
}

# ------------------------------------------------------------
# GPU EC2 INSTANCE
# ------------------------------------------------------------

resource "aws_instance" "gpu_worker" {
  count = var.gpu_enabled ? 1 : 0

  ami           = var.gpu_ami_id
  instance_type = var.gpu_instance_type

  subnet_id              = var.private_subnet_id
  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = var.instance_profile_name

  associate_public_ip_address = false

  root_block_device {
    volume_size = var.gpu_volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e

    mkdir -p /opt/physical-ai

    cat > /opt/physical-ai/runtime.env <<'ENV'
    ROBOT_MODE=simulation
    SIMULATION_ONLY=true
    PHYSICAL_ROBOT_CONTROL_ENABLED=false
    HARDWARE_ACTUATION_ENABLED=false
    ACTUATOR_COMMANDS_ENABLED=false
    REMOTE_MOTION_COMMANDS_ENABLED=false
    EMERGENCY_STOP_RESET_ENABLED=false
    ML_DEVICE=cuda
    ENV

    chmod 600 /opt/physical-ai/runtime.env
  EOF

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-gpu-worker"
    Role = "gpu-ml-inference"
    GPU  = "enabled"
  })
}

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

output "gpu_instance_id" {
  description = "GPU instance ID"
  value = var.gpu_enabled ? aws_instance.gpu_worker[0].id : null
}

output "gpu_private_ip" {
  description = "GPU private IP"
  value = var.gpu_enabled ? aws_instance.gpu_worker[0].private_ip : null
}

output "gpu_instance_type" {
  value = var.gpu_enabled ? var.gpu_instance_type : null
}
