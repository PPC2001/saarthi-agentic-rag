# Infrastructure as Code (IaC) — Saarthi AI AWS ECS Deployment

This directory contains production-ready Terraform Infrastructure as Code (IaC) for deploying **Saarthi AI** on **AWS ECS Fargate** with an Application Load Balancer (ALB), auto-scaling, ECR repository, and AWS Secrets Manager integration.

Modeled after enterprise cloud standards (similar to Cinépolis microservices), this setup provides separate environment configs for `DEV` and `PROD`.

---

## Architecture Overview

```
                      Internet
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   Application Load Balancer     │  (Public Subnets)
        │      (HTTP :80 / HTTPS :443)    │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │       Target Group (:8000)      │
        │      Health Check: /health      │
        └────────────────┬────────────────┘
                         │
          Private VPC Subnets (awsvpc)
        ┌─────────────────────────────────┐
        │   ECS Service (Fargate)         │
        │   ┌───────────────────────────┐ │
        │   │ Container: saarthi-ai     │ │
        │   │ Port: 8000                │ │
        │   │ Pulls Image from ECR      │ │
        │   │ Pulls Secrets from SM     │ │
        │   │ Push Logs to CloudWatch   │ │
        │   └───────────────────────────┘ │
        │         Auto-Scaling Target     │
        │    (CPU @ 70% | Memory @ 80%)   │
        └─────────────────────────────────┘
```

---

## File Structure

| File | Purpose |
|------|---------|
| [`terraform.tf`](./terraform.tf) | Terraform version, S3 remote state backend, DynamoDB lock, and AWS provider. |
| [`variable.tf`](./variable.tf) | Input variable definitions with types and descriptions. |
| [`local.tf`](./local.tf) | Local variables, environment variables mapping, secrets mapping, and resource tags. |
| [`main.tf`](./main.tf) | Complete AWS infrastructure: ECR, CloudWatch, ALB, Security Groups, IAM Roles, ECS Cluster, Task Definition, Service, and AutoScaling. |
| [`outputs.tf`](./outputs.tf) | Output attributes: ALB DNS Name, ECS Service, Cluster Name, and ECR URL. |
| [`tfvars.dev.json`](./tfvars.dev.json) | Variables & placeholders for **DEV** environment. |
| [`tfvars.prod.json`](./tfvars.prod.json) | Variables & placeholders for **PROD** environment (high-availability sizing). |
| [`.gitignore`](./.gitignore) | Ignores local `.terraform/` and state files while preserving environment configs. |

---

## AWS Details & Placeholder Reference Checklist

Before deploying, update the placeholders in the following files with your AWS account details:

### 1. In `terraform.tf`
- `<PLACEHOLDER_S3_TERRAFORM_STATE_BUCKET>`: Name of your S3 bucket storing Terraform state (e.g., `saarthi-ai-terraform-states`).
- `<PLACEHOLDER_DYNAMODB_LOCK_TABLE>`: Name of your DynamoDB table used for state locking (e.g., `saarthi-ai-terraform-locks`).
*(Note: If testing locally without remote state first, you can temporarily comment out the `backend "s3"` block).*

### 2. In `tfvars.dev.json` and `tfvars.prod.json`

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `account_id` | Your 12-digit AWS Account ID | `"123456789012"` |
| `aws_region` | AWS region where you deploy | `"us-east-1"` or `"ap-south-1"` |
| `vpc_id` | ID of your VPC | `"vpc-0123456789abcdef0"` |
| `private_subnet_ids` | IDs of private subnets for ECS tasks | `["subnet-01111111111111111", "subnet-02222222222222222"]` |
| `public_subnet_ids` | IDs of public subnets for the Internet-facing ALB | `["subnet-03333333333333333", "subnet-04444444444444444"]` |
| `certificate_arn` | (Optional) ACM Certificate ARN for HTTPS | `"arn:aws:acm:us-east-1:123456789012:certificate/xxxx"` (or `""` for HTTP only) |
| `QDRANT_HOST` | Internal hostname / endpoint of your Qdrant vector database | `"http://qdrant.dev.internal:6333"` |
| `CORS_ORIGINS` | Web frontend URL allowed to call the API | `"https://saarthi.yourdomain.com"` |

### 3. Application Secrets (in AWS Secrets Manager)
Create the following secrets in AWS Secrets Manager and place their ARNs under `task_definition.secrets`:
- `LOGFIRE_TOKEN`: Token for Pydantic Logfire observability.
- `QWEN_API_KEY`: API key for DashScope / Qwen LLM.
- `TAVILY_API_KEY`: API key for Tavily regulatory search.
- `QDRANT_API_KEY`: Secret key for Qdrant database access.
- `SECRET_KEY`: Random 64-char key for session encryption & JWT tokens.

---

## Deployment Instructions

### Step 1: Build & Push Docker Image to Amazon ECR

```bash
# 1. Log in to AWS ECR
aws ecr get-login-password --region <YOUR_AWS_REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<YOUR_AWS_REGION>.amazonaws.com

# 2. Build the production Docker image from the repository root
cd /home/pratik/projects/rag-with-aws
docker build -t saarthi-ai:latest -f Dockerfile .

# 3. Tag and push to ECR
docker tag saarthi-ai:latest <ACCOUNT_ID>.dkr.ecr.<YOUR_AWS_REGION>.amazonaws.com/saarthi-ai:latest
docker push <ACCOUNT_ID>.dkr.ecr.<YOUR_AWS_REGION>.amazonaws.com/saarthi-ai:latest
```

---

### Step 2: Deploy DEV Environment

```bash
cd /home/pratik/projects/rag-with-aws/.iac

# Initialize Terraform plugins & backend
terraform init

# Select or create the dev workspace
terraform workspace select dev || terraform workspace new dev

# Preview the changes
terraform plan -var-file=tfvars.dev.json

# Apply and provision infrastructure
terraform apply -var-file=tfvars.dev.json
```

---

### Step 3: Deploy PROD Environment

```bash
cd /home/pratik/projects/rag-with-aws/.iac

# Select or create the prod workspace
terraform workspace select prod || terraform workspace new prod

# Preview changes with prod sizing (min 2, desired 3, max 10 tasks)
terraform plan -var-file=tfvars.prod.json

# Apply to production
terraform apply -var-file=tfvars.prod.json
```

---

### Step 4: Verify Deployment

After `terraform apply` succeeds, inspect the outputs:
```bash
terraform output alb_dns_name
```
Then query the health check endpoint:
```bash
curl http://<ALB_DNS_NAME>/health
```
Expected response:
```json
{"status":"healthy","service":"saarthi-ai"}
```
