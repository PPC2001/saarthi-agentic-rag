variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "Target AWS Region for deployment"
}

variable "account_id" {
  type        = string
  description = "Target AWS Account ID (e.g. 123456789012)"
}

variable "environment_name" {
  type        = string
  description = "Target Environment Name (e.g. DEV, PROD)"
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
  description = "VPC ID where ECS cluster and ALB will be provisioned (e.g. vpc-xxxxxxxxxxxxxxxxx)"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "List of private subnet IDs for ECS Fargate tasks (e.g. ['subnet-xxxxxxxx', 'subnet-yyyyyyyy'])"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "List of public subnet IDs for Internet-facing ALB (e.g. ['subnet-aaaaaaaa', 'subnet-bbbbbbbb'])"
}

variable "container_port" {
  type        = number
  default     = 8000
  description = "Container port exposed by Saarthi AI FastAPI application"
}

variable "container_image" {
  type        = string
  default     = ""
  description = "Full Docker image URI in AWS ECR (e.g. <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/saarthi-ai:latest)"
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
    secrets      = map(string)
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
