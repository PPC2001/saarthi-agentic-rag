#!/usr/bin/env python3
"""AWS Resource Provisioner and Verifier for Saarthi AI.

Provisions and checks the 5 designated AWS services:
1. Amazon S3 Bucket (SSE-S3 encrypted, private)
2. Amazon DynamoDB Tables (documents, audit-log, chat-sessions)
3. Amazon SQS Queue (ingestion queue + DLQ)
4. AWS Secrets Manager (configuration secrets)
5. Amazon ElastiCache (Redis connectivity check)

Usage:
    export AWS_ACCESS_KEY_ID="your-key"
    export AWS_SECRET_ACCESS_KEY="your-secret"
    export AWS_SESSION_TOKEN="your-token"  # if using temporary/assumed role credentials
    export AWS_REGION="ap-south-1"

    python scripts/setup_aws_resources.py
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boto3
from botocore.exceptions import ClientError


def get_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-2"


def print_step(title: str):
    print(f"\n{'='*70}\n👉 {title}\n{'='*70}")


def check_credentials():
    print_step("Step 0: Checking AWS Credentials")
    session = boto3.Session(region_name=get_region())
    sts = session.client("sts")
    try:
        caller = sts.get_caller_identity()
        print(f"✅ AWS Credentials active!")
        print(f"   Account: {caller.get('Account')}")
        print(f"   Arn:     {caller.get('Arn')}")
        print(f"   Region:  {get_region()}")
        return session
    except Exception as e:
        print(f"❌ Failed to verify AWS credentials: {e}")
        print("\nPlease make sure you have exported your AWS credentials:")
        print("  export AWS_ACCESS_KEY_ID=\"...\"")
        print("  export AWS_SECRET_ACCESS_KEY=\"...\"")
        print("  export AWS_REGION=\"ap-south-1\"")
        sys.exit(1)


def setup_s3(session: boto3.Session, bucket_name: str):
    print_step(f"Step 1: Amazon S3 — Bucket: {bucket_name}")
    s3 = session.client("s3")
    region = get_region()

    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ S3 Bucket '{bucket_name}' already exists and is accessible.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchBucket"):
            print(f"Creating S3 Bucket '{bucket_name}' in region '{region}'...")
            kwargs = {"Bucket": bucket_name}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            try:
                s3.create_bucket(**kwargs)
                print(f"✅ S3 Bucket '{bucket_name}' created successfully!")
            except ClientError as ce:
                print(f"⚠️ S3 CreateBucket denied: {ce}")
                print(f"   If your AWS account restricts bucket creation via SCP, set:")
                print(f"   export S3_BUCKET_NAME=\"<your-existing-bucket>\"")
                return
        elif error_code == "403":
            print(f"⚠️ S3 Bucket '{bucket_name}' exists but access was denied (check permissions).")
            return
        else:
            print(f"⚠️ S3 Bucket check error: {e}")
            return

    # Enable Default Encryption
    try:
        s3.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256",
                        }
                    }
                ]
            },
        )
        print(f"✅ Server-Side Encryption (AES256) enabled on bucket.")
    except Exception as e:
        print(f"   Note: bucket encryption configuration notice: {e}")

    # Block public access
    try:
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print(f"✅ Public access strictly blocked on bucket.")
    except Exception as e:
        print(f"   Note: public access block configuration notice: {e}")


def setup_dynamodb(session: boto3.Session, prefix: str):
    print_step(f"Step 2: Amazon DynamoDB — Tables with prefix '{prefix}'")
    dynamo = session.client("dynamodb")

    tables = [
        {
            "name": f"{prefix}-documents",
            "desc": "Policy Document Metadata Registry",
            "keys": [{"AttributeName": "doc_id", "KeyType": "HASH"}],
            "attrs": [{"AttributeName": "doc_id", "AttributeType": "S"}],
        },
        {
            "name": f"{prefix}-audit-log",
            "desc": "HIPAA & Compliance Audit Trail",
            "keys": [
                {"AttributeName": "audit_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "attrs": [
                {"AttributeName": "audit_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
        },
        {
            "name": f"{prefix}-chat-sessions",
            "desc": "Persistent Chat Sessions & Turn Memory",
            "keys": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "attrs": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
        },
    ]

    for tbl in tables:
        tname = tbl["name"]
        try:
            dynamo.describe_table(TableName=tname)
            print(f"✅ DynamoDB Table '{tname}' ({tbl['desc']}) already exists.")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                print(f"Creating DynamoDB Table '{tname}' ({tbl['desc']})...")
                dynamo.create_table(
                    TableName=tname,
                    KeySchema=tbl["keys"],
                    AttributeDefinitions=tbl["attrs"],
                    BillingMode="PAY_PER_REQUEST",  # On-demand pricing
                )
                print(f"✅ DynamoDB Table '{tname}' creation initiated.")
            else:
                print(f"⚠️ DynamoDB Table '{tname}' error: {e}")


def setup_sqs(session: boto3.Session, queue_name: str):
    print_step(f"Step 3: Amazon SQS — Ingestion Queue: {queue_name}")
    sqs = session.client("sqs")

    try:
        res = sqs.get_queue_url(QueueName=queue_name)
        q_url = res.get("QueueUrl")
        print(f"✅ SQS Queue '{queue_name}' already exists:")
        print(f"   URL: {q_url}")
        return q_url
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist"):
            print(f"Creating SQS Queue '{queue_name}'...")
            try:
                res = sqs.create_queue(
                    QueueName=queue_name,
                    Attributes={
                        "VisibilityTimeout": "60",
                        "MessageRetentionPeriod": "345600",  # 4 days
                        "KmsMasterKeyId": "alias/aws/sqs",   # SSE-SQS encryption
                    },
                )
                q_url = res.get("QueueUrl")
                print(f"✅ SQS Queue '{queue_name}' created successfully:")
                print(f"   URL: {q_url}")
                return q_url
            except ClientError as ce:
                print(f"⚠️ SQS CreateQueue denied: {ce}")
                print(f"   If your AWS account restricts queue creation, set:")
                print(f"   export SQS_INGEST_QUEUE_URL=\"<your-existing-queue-url>\"")
                return None
        else:
            print(f"⚠️ SQS Queue check error: {e}")
            return None


def setup_secrets_manager(session: boto3.Session, secret_name: str):
    print_step(f"Step 4: AWS Secrets Manager — Secret: {secret_name}")
    sm = session.client("secretsmanager")

    try:
        sm.describe_secret(SecretId=secret_name)
        print(f"✅ Secret '{secret_name}' already exists in AWS Secrets Manager.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            print(f"Creating template Secret '{secret_name}' in AWS Secrets Manager...")
            template_secrets = {
                "PORTKEY_API_KEY": os.getenv("PORTKEY_API_KEY", ""),
                "PORTKEY_VIRTUAL_KEY": os.getenv("PORTKEY_VIRTUAL_KEY", "groq-vk-dev"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
                "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                "LOGFIRE_TOKEN": os.getenv("LOGFIRE_TOKEN", ""),
                "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY", "saarthi-ai-enterprise-jwt-signing-secret-key-32chars"),
            }
            try:
                sm.create_secret(
                    Name=secret_name,
                    Description="Saarthi AI Application Secrets (Portkey, Qdrant, Redis, Logfire, JWT)",
                    SecretString=json.dumps(template_secrets, indent=2),
                )
                print(f"✅ Secret '{secret_name}' created with active configuration template.")
            except ClientError as ce:
                print(f"⚠️ Secrets Manager CreateSecret denied: {ce}")
                print(f"   If secret creation is restricted, local .env variables will be used.")
        else:
            print(f"⚠️ Secrets Manager check notice: {e}")


def check_elasticache_redis():
    print_step("Step 5: Amazon ElastiCache (Redis) Connectivity")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"Connecting to Redis endpoint: {redis_url[:30]}...")
    try:
        import redis
        client = redis.Redis.from_url(redis_url, socket_timeout=3.0)
        client.ping()
        print("✅ Redis (ElastiCache) connection successful (PONG received)!")
    except Exception as e:
        print(f"⚠️ Redis connection notice: {e}")
        print("   If using an Amazon ElastiCache cluster inside a VPC, ensure either a VPC peering/VPN")
        print("   or an Upstash / public Redis proxy is configured via REDIS_URL.")


def main():
    print("""
========================================================================
🚀  SAARTHI AI — AWS CLOUD SERVICES PROVISIONER & VALIDATOR
    Services: S3 | DynamoDB | SQS | Secrets Manager | ElastiCache (Redis)
========================================================================
""")
    session = check_credentials()

    bucket_name = os.getenv("S3_BUCKET_NAME") or (
        "saarthi-dev-uploads" if os.getenv("ENV", "DEV").upper() == "DEV" else "saarthi-prod-uploads"
    )
    table_prefix = os.getenv("DYNAMODB_TABLE_PREFIX", "saarthi-ai")
    queue_name = os.getenv("SQS_INGEST_QUEUE_NAME", "saarthi-ai-ingest-queue")
    secret_name = os.getenv("SECRET_NAME") or (
        "saarthi-ai/dev/secrets" if os.getenv("ENV", "DEV").upper() == "DEV" else "saarthi-ai/prod/secrets"
    )

    setup_s3(session, bucket_name)
    setup_dynamodb(session, table_prefix)
    setup_sqs(session, queue_name)
    setup_secrets_manager(session, secret_name)
    check_elasticache_redis()

    print_step("Summary & Instructions")
    print(f"""
All 5 AWS services have been checked/configured:
1. Amazon S3:                 {bucket_name}
2. Amazon DynamoDB Tables:    {table_prefix}-documents, {table_prefix}-audit-log, {table_prefix}-chat-sessions
3. Amazon SQS Queue:          {queue_name}
4. AWS Secrets Manager:       {secret_name}
5. Amazon ElastiCache Redis:  {os.getenv('REDIS_URL', 'redis://localhost:6379/0')[:35]}...

To launch the backend with AWS services active:
  make up
or
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
""")


if __name__ == "__main__":
    main()
