locals {
  name = "${var.service_name}-${lower(var.environment_name)}"

  # Default container image if not explicitly set
  container_image = var.container_image != "" ? var.container_image : "${var.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.service_name}:${var.git_commit_hash}"

  # Fallback to default VPC and subnets if not explicitly specified
  vpc_id             = var.vpc_id != "" ? var.vpc_id : try(data.aws_vpc.default[0].id, "")
  public_subnet_ids  = length(var.public_subnet_ids) > 0 ? var.public_subnet_ids : try(data.aws_subnets.default[0].ids, [])
  private_subnet_ids = length(var.private_subnet_ids) > 0 ? var.private_subnet_ids : local.public_subnet_ids

  # When tasks run in public/default subnets (no NAT gateway), assign_public_ip must be true to pull from ECR
  assign_public_ip = var.assign_public_ip != null ? var.assign_public_ip : (
    length(var.private_subnet_ids) == 0 || length(var.private_subnet_ids) == length(local.public_subnet_ids)
  )

  # Merged application runtime environments
  environments = merge(
    {
      SERVICE_NAME       = var.service_name
      ENVIRONMENT_NAME   = var.environment_name
      ENV                = var.environment_name
      AWS_REGION         = var.aws_region
      AWS_DEFAULT_REGION = var.aws_region
      PORT               = tostring(var.container_port)
    },
    var.task_definition.environments
  )

  # Secrets mapping: filter out empty or unresolved placeholder strings
  secrets = var.task_definition.secrets != null ? var.task_definition.secrets : {}

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
    if v != "" && !can(regex("<.*>", v))
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
      Region      = var.aws_region
      iac_source  = var.service_name
    }
  )
}
