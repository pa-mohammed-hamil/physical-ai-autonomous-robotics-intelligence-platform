# outputs: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# Physical AI Robotics - Terraform Outputs
# File: terraform/outputs.tf
# ============================================================

# Networking
output "network_vpc_id" {
  value = aws_vpc.main.id
}

output "network_public_subnets" {
  value = aws_subnet.public[*].id
}

output "network_private_subnets" {
  value = aws_subnet.private[*].id
}

output "nat_gateway_id" {
  value = aws_nat_gateway.main.id
}

# EC2 Compute
output "backend_instance_id" {
  value = aws_instance.backend.id
}

output "backend_instance_private_ip" {
  value = aws_instance.backend.private_ip
}

output "ml_instance_id" {
  value = aws_instance.ml.id
}

output "ml_instance_private_ip" {
  value = aws_instance.ml.private_ip
}

output "worker_instance_id" {
  value = aws_instance.worker.id
}

output "worker_instance_private_ip" {
  value = aws_instance.worker.private_ip
}

# GPU Compute
output "gpu_worker_instance_id" {
  value = try(aws_instance.gpu_worker[0].id, null)
}

output "gpu_worker_private_ip" {
  value = try(aws_instance.gpu_worker[0].private_ip, null)
}

# Database
output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "postgres_secret_arn" {
  value = aws_db_instance.postgres.master_user_secret[0].secret_arn
}

# Messaging
output "tasks_queue_url" {
  value = aws_sqs_queue.robot_tasks.url
}

output "alerts_queue_url" {
  value = aws_sqs_queue.robot_alerts.url
}

output "notifications_topic_arn" {
  value = aws_sns_topic.robot_notifications.arn
}

# Monitoring
output "cloudwatch_backend_log_group" {
  value = aws_cloudwatch_log_group.backend.name
}

output "cloudwatch_ml_log_group" {
  value = aws_cloudwatch_log_group.ml.name
}

output "cloudwatch_workers_log_group" {
  value = aws_cloudwatch_log_group.workers.name
}
