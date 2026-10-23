# storage: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# Physical AI Robotics - AWS Storage
# S3 | Model Artifacts | Datasets | Logs | Backups
# ============================================================

resource "aws_s3_bucket" "datasets" {
  bucket_prefix = "${var.project_name}-${var.environment}-datasets-"

  tags = local.common_tags
}

resource "aws_s3_bucket" "models" {
  bucket_prefix = "${var.project_name}-${var.environment}-models-"

  tags = local.common_tags
}

resource "aws_s3_bucket" "backups" {
  bucket_prefix = "${var.project_name}-${var.environment}-backups-"

  tags = local.common_tags
}

resource "aws_s3_bucket" "logs" {
  bucket_prefix = "${var.project_name}-${var.environment}-logs-"

  tags = local.common_tags
}

# Block public access to all buckets
resource "aws_s3_bucket_public_access_block" "private" {
  for_each = {
    datasets = aws_s3_bucket.datasets.id
    models   = aws_s3_bucket.models.id
    backups  = aws_s3_bucket.backups.id
    logs     = aws_s3_bucket.logs.id
  }

  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for models and backups
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "encrypted" {
  for_each = {
    datasets = aws_s3_bucket.datasets.id
    models   = aws_s3_bucket.models.id
    backups  = aws_s3_bucket.backups.id
    logs     = aws_s3_bucket.logs.id
  }

  bucket = each.value

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Expire temporary logs after 30 days
resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    filter {}

    expiration {
      days = 30
    }
  }
}

# Outputs
output "datasets_bucket_name" {
  value = aws_s3_bucket.datasets.id
}

output "models_bucket_name" {
  value = aws_s3_bucket.models.id
}

output "backups_bucket_name" {
  value = aws_s3_bucket.backups.id
}

output "logs_bucket_name" {
  value = aws_s3_bucket.logs.id
}
