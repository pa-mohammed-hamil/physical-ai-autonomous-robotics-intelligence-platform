# monitoring: infrastructure placeholder. No cloud resources are provisioned.

# Physical AI Robotics - AWS Monitoring
# CloudWatch Logs, Metrics and Alarms

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/physical-ai/${var.environment}/backend"
  retention_in_days = 14

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "ml" {
  name              = "/physical-ai/${var.environment}/ml"
  retention_in_days = 14

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "workers" {
  name              = "/physical-ai/${var.environment}/workers"
  retention_in_days = 14

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "backend_cpu" {
  alarm_name          = "${var.project_name}-backend-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80

  dimensions = {
    InstanceId = aws_instance.backend.id
  }

  alarm_actions = [aws_sns_topic.robot_notifications.arn]

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "ml_cpu" {
  alarm_name          = "${var.project_name}-ml-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 85

  dimensions = {
    InstanceId = aws_instance.ml.id
  }

  alarm_actions = [aws_sns_topic.robot_notifications.arn]

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${var.project_name}-database-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.identifier
  }

  alarm_actions = [aws_sns_topic.robot_notifications.arn]

  tags = local.common_tags
}

output "backend_log_group" {
  value = aws_cloudwatch_log_group.backend.name
}

output "ml_log_group" {
  value = aws_cloudwatch_log_group.ml.name
}

output "workers_log_group" {
  value = aws_cloudwatch_log_group.workers.name
}
