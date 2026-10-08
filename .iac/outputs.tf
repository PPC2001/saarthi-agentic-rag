output "alb_dns_name" {
  description = "The public DNS name of the Application Load Balancer to access Saarthi AI"
  value       = aws_lb.this.dns_name
}

output "ecs_cluster_name" {
  description = "The name of the ECS Cluster"
  value       = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  description = "The name of the ECS Service"
  value       = aws_ecs_service.this.name
}

output "task_definition_arn" {
  description = "The ARN of the ECS Task Definition"
  value       = aws_ecs_task_definition.this.arn
}

output "ecr_repository_url" {
  description = "The Amazon ECR Docker repository URL for Saarthi AI images"
  value       = aws_ecr_repository.this.repository_url
}

output "cloudwatch_log_group" {
  description = "CloudWatch Log Group name for container logs"
  value       = aws_cloudwatch_log_group.this.name
}
