# database: infrastructure placeholder. No cloud resources are provisioned.

# ============================================================
# PHYSICAL AI ROBOTICS - AWS DATABASE
# File: terraform/database.tf
# PostgreSQL 16 | AWS RDS | Private Networking
# ============================================================

variable "database_subnet_ids" {
  description = "At least two private subnet IDs in different AZs"
  type        = list(string)
}

variable "database_name" {
  type    = string
  default = "physical_ai"
}

variable "database_username" {
  type    = string
  default = "robotics"
}

variable "database_instance_class" {
  type    = string
  default = "db.t3.micro"
}

# ------------------------------------------------------------
# RDS SUBNET GROUP
# ------------------------------------------------------------

resource "aws_db_subnet_group" "postgres" {
  name       = "${var.project_name}-${var.environment}-db-subnets"
  subnet_ids = var.database_subnet_ids

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-db-subnets"
  })
}

# ------------------------------------------------------------
# DATABASE SECURITY GROUP
# ------------------------------------------------------------

resource "aws_security_group" "postgres" {
  name_prefix = "${var.project_name}-postgres-"
  description = "PostgreSQL access from application instances"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL from application compute"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.compute.id]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-postgres-sg"
  })
}

# ------------------------------------------------------------
# POSTGRESQL RDS INSTANCE
# ------------------------------------------------------------

resource "aws_db_instance" "postgres" {
  identifier = "${var.project_name}-${var.environment}-postgres"

  engine         = "postgres"
  engine_version = "16"

  instance_class = var.database_instance_class

  db_name  = var.database_name
  username = var.database_username

  # AWS generates and manages the master password.
  manage_master_user_password = true

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]

  publicly_accessible = false
  multi_az            = false

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  auto_minor_version_upgrade = true
  deletion_protection        = true
  skip_final_snapshot        = false
  final_snapshot_identifier  = "${var.project_name}-${var.environment}-final-snapshot"

  copy_tags_to_snapshot = true

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-postgres"
    Role = "application-database"
  })
}

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

output "database_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "database_address" {
  value = aws_db_instance.postgres.address
}

output "database_port" {
  value = aws_db_instance.postgres.port
}

output "database_name" {
  value = aws_db_instance.postgres.db_name
}

output "database_master_secret_arn" {
  value = aws_db_instance.postgres.master_user_secret[0].secret_arn
}
