# messaging: infrastructure placeholder. No cloud resources are provisioned.

# Physical AI Robotics - AWS Messaging
# Amazon SQS for background tasks and robot events

resource "aws_sqs_queue" "robot_tasks" {
  name                       = "${var.project_name}-${var.environment}-tasks"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 345600

  tags = local.common_tags
}

resource "aws_sqs_queue" "robot_alerts" {
  name                      = "${var.project_name}-${var.environment}-alerts"
  message_retention_seconds = 604800

  tags = local.common_tags
}

resource "aws_sns_topic" "robot_notifications" {
  name = "${var.project_name}-${var.environment}-notifications"

  tags = local.common_tags
}

output "robot_tasks_queue_url" {
  value = aws_sqs_queue.robot_tasks.url
}

output "robot_alerts_queue_url" {
  value = aws_sqs_queue.robot_alerts.url
}

output "robot_notifications_topic_arn" {
  value = aws_sns_topic.robot_notifications.arn
}
