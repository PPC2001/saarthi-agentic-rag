#!/usr/bin/env python3
"""Run SQS Background Ingestion Worker for Saarthi AI."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.core.aws.sqs_worker import run_sqs_worker

if __name__ == "__main__":
    print("Starting Saarthi AI Amazon SQS Ingestion Worker...")
    asyncio.run(run_sqs_worker())
