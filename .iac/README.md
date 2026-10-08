# Infrastructure as Code (IaC) — Saarthi AI AWS ECS Deployment
**Target Region:** `ap-southeast-2` (Sydney)  
**Target AWS Account:** `974913402908`  
**Environment:** `DEV`

This directory contains production-ready Terraform Infrastructure as Code (IaC) for deploying **Saarthi AI** on **AWS ECS Fargate** with an Application Load Balancer (ALB), auto-scaling, ECR repository, and AWS Secrets Manager integration.

---

## Architecture Overview

```
                      Internet
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   Application Load Balancer     │  (Public Subnets in ap-southeast-2)
        │      (HTTP :80 / HTTPS :443)    │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │       Target Group (:8000)      │
        │      Health Check: /health      │
        └────────────────┬────────────────┘
                         │
          VPC Subnets (awsvpc, ap-southeast-2)
        ┌─────────────────────────────────┐
        │   ECS Service (Fargate)         │
        │   ┌───────────────────────────┐ │
        │   │ Container: saarthi-ai     │ │
        │   │ Port: 8000                │ │
        │   │ Pulls Image from ECR      │ │
        │   │ Task Role: DynamoDB, S3,  │ │
        │   │ Secrets Manager, Bedrock  │ │
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
| [`terraform.tf`](./terraform.tf) | Terraform version, AWS provider (`ap-southeast-2`), and state backend. |
| [`variable.tf`](./variable.tf) | Input variables configured for account `974913402908` and `ap-southeast-2`. |
| [`local.tf`](./local.tf) | Dynamic default VPC discovery, ECR image resolution, and environment merging. |
| [`main.tf`](./main.tf) | AWS infrastructure: ECR, CloudWatch, ALB, Security Groups, IAM Roles, ECS Cluster, Task Definition, Service, and AutoScaling. |
| [`outputs.tf`](./outputs.tf) | Outputs: ALB DNS Name, ECS Service, Cluster Name, and ECR URL. |
| [`tfvars.dev.json`](./tfvars.dev.json) | Variables pre-configured for **DEV** in `ap-southeast-2` (`974913402908`). |
| [`tfvars.prod.json`](./tfvars.prod.json) | Variables template for **PROD** environment. |

---

## Pre-Requisites & Configuration

1. **AWS CLI Authenticated:**
   ```bash
   aws sts get-caller-identity
   # Should return Account: 974913402908
   ```
2. **Target Region:** `ap-southeast-2`
3. **Secrets Manager Secret:**
   Ensure secret `saarthi-ai/dev/secrets` exists in AWS Secrets Manager (`ap-southeast-2`) with your API keys (`PORTKEY_API_KEY`, `LOGFIRE_TOKEN`, `QDRANT_API_KEY`, `JWT_SECRET_KEY`). If not created yet, run:
   ```bash
   python scripts/setup_aws_resources.py
   ```

---

## Deployment Walkthrough

### Step 1: Create ECR Repository & Push Docker Image

```bash
# 1. Authenticate Docker with Amazon ECR in ap-southeast-2
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 974913402908.dkr.ecr.ap-southeast-2.amazonaws.com

# 2. Ensure ECR repository exists
aws ecr describe-repositories --repository-names saarthi-ai --region ap-southeast-2 || \
  aws ecr create-repository --repository-name saarthi-ai --region ap-southeast-2

# 3. Build the container from saarthi-ai directory
cd /home/pratik/projects/rag-with-aws/saarthi-ai
docker build -t saarthi-ai:latest -f Dockerfile .

# 4. Tag and push to ECR
docker tag saarthi-ai:latest 974913402908.dkr.ecr.ap-southeast-2.amazonaws.com/saarthi-ai:latest
docker push 974913402908.dkr.ecr.ap-southeast-2.amazonaws.com/saarthi-ai:latest
```

---

### Step 2: Deploy Infrastructure via Terraform

```bash
cd /home/pratik/projects/rag-with-aws/saarthi-ai/.iac

# 1. Initialize Terraform
terraform init

# 2. Preview the plan
terraform plan -var-file=tfvars.dev.json

# 3. Apply and deploy ECS Cluster, ALB & Service
terraform apply -var-file=tfvars.dev.json
```

---

### Step 3: Verify Deployment

After `terraform apply` finishes, it will print the outputs:
```bash
Apply complete! Resources: 18 added, 0 changed, 0 destroyed.

Outputs:
alb_dns_name = "saarthi-ai-dev-alb-xxxxxxxx.ap-southeast-2.elb.amazonaws.com"
ecr_repository_url = "974913402908.dkr.ecr.ap-southeast-2.amazonaws.com/saarthi-ai"
ecs_cluster_name = "saarthi-ai-dev-cluster"
ecs_service_name = "saarthi-ai-dev"
```

Verify the health check:
```bash
curl http://<alb_dns_name>/health
# {"status":"ok","version":"1.0.0","environment":"DEV"}
```

Use `http://<alb_dns_name>` as the `VITE_API_URL` in your Vercel deployment of [`saarthi-ai-portal`](https://github.com/PPC2001/saarthi-ai-portal).
