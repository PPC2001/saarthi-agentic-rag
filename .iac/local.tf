locals {
  name = "${var.service_name}-${lower(var.environment_name)}"

  # Default container image if not explicitly set
  container_image = var.container_image != "" ? var.container_image : "${var.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.service_name}:${var.git_commit_hash}"

  # Merged application runtime environments
  environments = merge(
    {
      SERVICE_NAME       = var.service_name
      ENVIRONMENT_NAME   = var.environment_name
      AWS_DEFAULT_REGION = var.aws_region
      PORT               = tostring(var.container_port)
    },
    var.task_definition.environments
  )

  # Merged secrets mapping: environment variable name -> Secrets Manager ARN
  secrets = var.task_definition.secrets

  # Formatted for container_definitions JSON
  container_environments = [
    for k, v in local.environments : {
      name  = k
      value = v
    }
  ]

  container_secrets = [
    for k, v in local.secrets : {
      name      = k
      valueFrom = v
    }
  ]

  # Common resource tags
  tags = merge(
    var.tags,
    {
      Service     = var.service_name
      Environment = var.environment_name
      GitCommit   = var.git_commit_hash
      GitTag      = var.git_tag_name
      ManagedBy   = "Terraform"
      iac_source  = var.service_name
    }
  )
}
