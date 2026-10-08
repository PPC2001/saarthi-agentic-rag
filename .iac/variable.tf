variable "aws_region" {
  type        = string
  default     = "ap-southeast-2"
  description = "Target AWS Region for deployment (Sydney)"
}

variable "account_id" {
  type        = string
  default     = "974913402908"
  description = "Target AWS Account ID (974913402908)"
}

variable "environment_name" {
  type        = string
  default     = "DEV"
  description = "Target Environment Name (DEV or PROD)"
}

variable "service_name" {
  type        = string
  default     = "saarthi-ai"
  description = "Logical name for the microservice / ECS workload"
}

variable "git_commit_hash" {
  type        = string
  default     = "latest"
  description = "Git Commit SHA for deployment traceability and container tagging"
}

variable "git_tag_name" {
  type        = string
  default     = "latest"
  description = "Git Release Tag (e.g. v1.0.0 or latest)"
}

variable "vpc_id" {
  type        = string
  default     = ""
  description = "VPC ID where ECS cluster and ALB will be provisioned. If empty, the default VPC is used."
}

variable "private_subnet_ids" {
  type        = list(string)
  default     = []
  description = "List of private subnet IDs for ECS Fargate tasks. If empty, default subnets are used."
}

variable "public_subnet_ids" {
  type        = list(string)
  default     = []
  description = "List of public subnet IDs for Internet-facing ALB. If empty, default subnets are used."
}

variable "assign_public_ip" {
  type        = bool
  default     = null
  description = "Whether to assign a public IP to Fargate tasks. Defaults to true when running in default/public subnets."
}

variable "container_port" {
  type        = number
  default     = 8000
  description = "Container port exposed by Saarthi AI FastAPI application"
}

variable "container_image" {
  type        = string
  default     = ""
  description = "Full Docker image URI in AWS ECR (e.g. 974913402908.dkr.ecr.ap-southeast-2.amazonaws.com/saarthi-ai:latest)"
}

variable "certificate_arn" {
  type        = string
  default     = ""
  description = "Optional ACM SSL/TLS certificate ARN for HTTPS listener on the ALB"
}

variable "alb_arn" {
  type        = string
  default     = ""
  description = "Optional ARN of a pre-existing ALB if reusing shared corporate ingress"
}

variable "task_definition" {
  type = object({
    cpu          = number
    memory       = number
    environments = map(string)
    secrets      = optional(map(string), {})
  })
  description = "ECS Task Definition CPU, Memory, environment variables, and Secrets Manager ARNs"
}

variable "service" {
  type = object({
    max_capacity  = number
    desired_count = number
    min_capacity  = number
  })
  description = "ECS Service scaling capacities (min_capacity, desired_count, max_capacity)"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Resource tags to apply across all provisioned infrastructure"
}
