terraform {
  required_version = ">= 1.5.0"

  # ============================================================================
  # S3 REMOTE STATE BACKEND
  # Replace placeholders with your remote state S3 bucket & DynamoDB table.
  # If running locally for initial test, you can temporarily comment out the
  # backend block.
  # ============================================================================
  backend "s3" {
    bucket         = "<PLACEHOLDER_S3_TERRAFORM_STATE_BUCKET>" # e.g. saarthi-terraform-states-dev
    key            = "saarthi-ai/terraform.tfstate"
    region         = "us-east-1"                         # e.g. us-east-1 or ap-south-1
    dynamodb_table = "<PLACEHOLDER_DYNAMODB_LOCK_TABLE>" # e.g. saarthi-terraform-locks
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.94.1"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # ============================================================================
  # CROSS-ACCOUNT / DEPLOYMENT ASSUME ROLE (OPTIONAL)
  # Uncomment assume_role if deploying via CI/CD cross-account role like Cinépolis.
  # Otherwise leave commented out to use your default AWS CLI / IAM credentials.
  # ============================================================================
  # assume_role {
  #   role_arn     = "arn:aws:iam::${var.account_id}:role/${local.name}-deployer-role"
  #   session_name = "${local.name}-${terraform.workspace}"
  # }

  default_tags {
    tags = local.tags
  }
}
