"""AWS Services integration module for Saarthi AI.

Provides unified clients and utilities for:
- Amazon ElastiCache (Redis)
- Amazon SQS
- AWS Secrets Manager
- Amazon S3
- Amazon DynamoDB
"""

from app.core.aws.dynamodb_service import DynamoDBService, get_dynamodb_service
from app.core.aws.s3_service import S3Service, get_s3_service
from app.core.aws.secrets_manager import fetch_and_apply_secrets
from app.core.aws.session import get_aioboto3_session, get_boto3_session, has_aws_credentials
from app.core.aws.sqs_service import SQSService, get_sqs_service

__all__ = [
    "has_aws_credentials",
    "get_boto3_session",
    "get_aioboto3_session",
    "S3Service",
    "get_s3_service",
    "DynamoDBService",
    "get_dynamodb_service",
    "SQSService",
    "get_sqs_service",
    "fetch_and_apply_secrets",
]
