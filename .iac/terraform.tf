terraform {
  required_version = ">= 1.5.0"

  # ============================================================================
  # S3 REMOTE STATE BACKEND (OPTIONAL)
  # Uncomment and configure once your remote state S3 bucket & lock table are created:
  # ============================================================================
  # backend "s3" {
  #   bucket         = "saarthi-terraform-states-974913402908"
  #   key            = "saarthi-ai/dev/terraform.tfstate"
  #   region         = "ap-southeast-2"
  #   dynamodb_table = "saarthi-terraform-locks"
  #   encrypt        = true
  # }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.94.1"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}
