"""AWS DynamoDB & Redis State Checkpointer for LangGraph.

Provides durable, cloud-native conversational state persistence across multi-turn
RAG dialogues without requiring heavyweight RDS PostgreSQL instances:
1. `AsyncDynamoDBSaver`: Serverless DynamoDB checkpointer with native TTL auto-purging.
2. `AsyncRedisSaver`: Sub-millisecond in-memory session checkpointer.
3. `get_checkpointer`: Factory adhering to Open-Closed and Dependency Inversion principles.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from redis.asyncio import Redis

from app.config import Settings
from app.core.aws.session import get_aioboto3_session, get_aws_region, has_aws_credentials
from app.observability.logging import get_logger

logger = get_logger("langgraph.checkpointer")


def _encode_typed(dumped: tuple[str, bytes]) -> dict[str, str]:
    """Encode typed serialization tuple safely for DynamoDB / JSON."""
    type_str, data_bytes = dumped
    return {
        "t": type_str,
        "d": base64.b64encode(data_bytes).decode("ascii"),
    }


def _decode_typed(payload: dict[str, str] | str) -> tuple[str, bytes]:
    """Decode safely back into typed serialization tuple."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload["t"], base64.b64decode(payload["d"].encode("ascii"))


class AsyncDynamoDBSaver(BaseCheckpointSaver[str]):
    """Cloud-native LangGraph checkpointer backed by Amazon DynamoDB.

    Key Features:
    - Zero idle cost (AWS On-Demand serverless pricing).
    - Native TTL auto-expiring old state snapshots (IRDAI compliance hygiene).
    - In-memory hot cache for sub-millisecond sequential graph invocations.
    - Graceful fail-open fallback if AWS credentials are not configured.
    """

    def __init__(
        self,
        table_name: str = "saarthi-ai-chat-sessions",
        ttl_days: int = 30,
        region_name: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.table_name = table_name
        self.ttl_seconds = ttl_days * 86400
        self.region_name = region_name or get_aws_region()
        self._memory = InMemorySaver(serde=self.serde)

    def is_aws_available(self) -> bool:
        """Verify whether AWS credentials are active."""
        return has_aws_credentials()

    # ── 1. Retrieval (aget_tuple) ─────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch latest checkpoint for a thread_id from DynamoDB or hot cache."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        # 1. Check local memory cache first for lowest latency
        mem_tuple = await self._memory.aget_tuple(config)
        if mem_tuple:
            return mem_tuple

        if not self.is_aws_available():
            return None

        try:
            from boto3.dynamodb.conditions import Key

            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_name)

                item: dict[str, Any] | None = None
                if checkpoint_id:
                    sort_key = f"ckpt#{checkpoint_ns}#{checkpoint_id}"
                    res = await table.get_item(
                        Key={"session_id": thread_id, "timestamp": sort_key}
                    )
                    item = res.get("Item")
                else:
                    # Fetch latest checkpoint for this namespace
                    prefix = f"ckpt#{checkpoint_ns}#"
                    res = await table.query(
                        KeyConditionExpression=(
                            Key("session_id").eq(thread_id)
                            & Key("timestamp").begins_with(prefix)
                        ),
                        ScanIndexForward=False,  # latest first
                        Limit=1,
                    )
                    items = res.get("Items", [])
                    item = items[0] if items else None

                if not item:
                    return None

                ckpt_raw = _decode_typed(item["checkpoint_payload"])
                meta_raw = _decode_typed(item["metadata_payload"])

                checkpoint_: Checkpoint = self.serde.loads_typed(ckpt_raw)
                metadata_: CheckpointMetadata = self.serde.loads_typed(meta_raw)
                actual_ckpt_id: str = item["checkpoint_id"]
                parent_id: str | None = item.get("parent_checkpoint_id")

                # Load channel blobs for versioned state
                channel_values: dict[str, Any] = {}
                channel_versions: ChannelVersions = checkpoint_.get("channel_versions", {})
                for ch, ver in channel_versions.items():
                    blob_key = f"blob#{checkpoint_ns}#{ch}#{ver}"
                    b_res = await table.get_item(
                        Key={"session_id": thread_id, "timestamp": blob_key}
                    )
                    b_item = b_res.get("Item")
                    if b_item and "payload" in b_item:
                        b_raw = _decode_typed(b_item["payload"])
                        if b_raw[0] != "empty":
                            channel_values[ch] = self.serde.loads_typed(b_raw)

                # Load pending writes
                write_prefix = f"write#{actual_ckpt_id}#"
                w_res = await table.query(
                    KeyConditionExpression=(
                        Key("session_id").eq(thread_id)
                        & Key("timestamp").begins_with(write_prefix)
                    )
                )
                pending_writes = []
                for w_item in w_res.get("Items", []):
                    w_raw = _decode_typed(w_item["payload"])
                    pending_writes.append(
                        (
                            w_item["task_id"],
                            w_item["channel"],
                            self.serde.loads_typed(w_raw),
                        )
                    )

                ckpt_tuple = CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": actual_ckpt_id,
                        }
                    },
                    checkpoint={
                        **checkpoint_,
                        "channel_values": channel_values,
                    },
                    metadata=metadata_,
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_id,
                            }
                        }
                        if parent_id
                        else None
                    ),
                    pending_writes=pending_writes,
                )

                # Update local memory hot cache
                try:
                    await self._memory.aput(
                        config=ckpt_tuple.config,
                        checkpoint=ckpt_tuple.checkpoint,
                        metadata=ckpt_tuple.metadata,
                        new_versions=channel_versions,
                    )
                except Exception:
                    pass

                return ckpt_tuple

        except Exception as e:
            logger.warning(
                "DYNAMODB_CHECKPOINT_GET_FAILED",
                thread_id=thread_id,
                error=str(e),
            )
            return await self._memory.aget_tuple(config)

    # ── 2. Persistence (aput) ─────────────────────────────────────────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store graph checkpoint, metadata, and channel blobs in DynamoDB."""
        # 1. Always update local memory cache first
        updated_config = await self._memory.aput(config, checkpoint, metadata, new_versions)

        if not self.is_aws_available():
            return updated_config

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = checkpoint["id"]
        ttl_epoch: int = int(time.time()) + self.ttl_seconds
        parent_id = config["configurable"].get("checkpoint_id")

        try:
            c = checkpoint.copy()
            values: dict[str, Any] = c.pop("channel_values", {})

            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_name)

                # A. Store channel blobs for changed versions
                for k, v in new_versions.items():
                    blob_typed = (
                        self.serde.dumps_typed(values[k])
                        if k in values
                        else ("empty", b"")
                    )
                    blob_item = {
                        "session_id": thread_id,
                        "timestamp": f"blob#{checkpoint_ns}#{k}#{v}",
                        "checkpoint_type": "blob",
                        "channel": k,
                        "version": str(v),
                        "payload": _encode_typed(blob_typed),
                        "expires_at": ttl_epoch,
                    }
                    await table.put_item(Item=blob_item)

                # B. Store main checkpoint metadata
                meta_to_store = get_checkpoint_metadata(config, metadata)
                ckpt_item = {
                    "session_id": thread_id,
                    "timestamp": f"ckpt#{checkpoint_ns}#{checkpoint_id}",
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": checkpoint_ns,
                    "parent_checkpoint_id": parent_id or "",
                    "checkpoint_type": "checkpoint",
                    "checkpoint_payload": _encode_typed(self.serde.dumps_typed(c)),
                    "metadata_payload": _encode_typed(self.serde.dumps_typed(meta_to_store)),
                    "expires_at": ttl_epoch,
                    "updated_at": str(time.time()),
                }
                await table.put_item(Item=ckpt_item)

        except Exception as e:
            logger.warning(
                "DYNAMODB_CHECKPOINT_PUT_FAILED",
                thread_id=thread_id,
                ckpt_id=checkpoint_id,
                error=str(e),
            )

        return updated_config

    # ── 3. Writes (aput_writes) ───────────────────────────────────────────────

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending task writes in DynamoDB."""
        await self._memory.aput_writes(config, writes, task_id, task_path)

        if not self.is_aws_available():
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_id: str = config["configurable"]["checkpoint_id"]
        ttl_epoch: int = int(time.time()) + self.ttl_seconds

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_name)
                for idx, (c, v) in enumerate(writes):
                    write_typed = self.serde.dumps_typed(v)
                    item = {
                        "session_id": thread_id,
                        "timestamp": f"write#{checkpoint_id}#{task_id}#{WRITES_IDX_MAP.get(c, idx)}",
                        "checkpoint_type": "write",
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "channel": c,
                        "payload": _encode_typed(write_typed),
                        "expires_at": ttl_epoch,
                    }
                    await table.put_item(Item=item)
        except Exception as e:
            logger.debug("DYNAMODB_CHECKPOINT_PUT_WRITES_FAILED", error=str(e))

    # ── 4. History (alist) ────────────────────────────────────────────────────

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching the filter criteria."""
        async for item in self._memory.alist(config, filter=filter, before=before, limit=limit):
            yield item

    # ── 5. Cleanup (adelete_thread) ───────────────────────────────────────────

    async def adelete_thread(self, thread_id: str) -> None:
        """Purge thread history from memory and DynamoDB."""
        await self._memory.adelete_thread(thread_id)
        if not self.is_aws_available():
            return

        try:
            from boto3.dynamodb.conditions import Key

            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_name)
                res = await table.query(
                    KeyConditionExpression=Key("session_id").eq(thread_id)
                )
                items = res.get("Items", [])
                for it in items:
                    await table.delete_item(
                        Key={"session_id": thread_id, "timestamp": it["timestamp"]}
                    )
        except Exception as e:
            logger.warning("DYNAMODB_DELETE_THREAD_FAILED", thread_id=thread_id, error=str(e))


class AsyncRedisSaver(BaseCheckpointSaver[str]):
    """Sub-millisecond LangGraph checkpointer backed by Redis."""

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = 86400,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self._memory = InMemorySaver(serde=self.serde)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch checkpoint from Redis."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if not checkpoint_id:
            # Look up latest pointer
            latest_id = await self.redis.get(f"saarthi:ckpt:{thread_id}:{checkpoint_ns}:latest")
            if not latest_id:
                return await self._memory.aget_tuple(config)
            checkpoint_id = latest_id

        key = f"saarthi:ckpt:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        val = await self.redis.get(key)
        if not val:
            return await self._memory.aget_tuple(config)

        data = json.loads(val)
        ckpt_raw = _decode_typed(data["checkpoint"])
        meta_raw = _decode_typed(data["metadata"])

        checkpoint_ = self.serde.loads_typed(ckpt_raw)
        metadata_ = self.serde.loads_typed(meta_raw)

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint_,
            metadata=metadata_,
            parent_config=None,
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist checkpoint to Redis with TTL."""
        updated = await self._memory.aput(config, checkpoint, metadata, new_versions)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = checkpoint["id"]

        key = f"saarthi:ckpt:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        meta_to_store = get_checkpoint_metadata(config, metadata)

        payload = {
            "checkpoint": _encode_typed(self.serde.dumps_typed(checkpoint)),
            "metadata": _encode_typed(self.serde.dumps_typed(meta_to_store)),
        }

        await self.redis.set(key, json.dumps(payload), ex=self.ttl_seconds)
        await self.redis.set(
            f"saarthi:ckpt:{thread_id}:{checkpoint_ns}:latest",
            checkpoint_id,
            ex=self.ttl_seconds,
        )
        return updated

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self._memory.aput_writes(config, writes, task_id, task_path)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for it in self._memory.alist(config, filter=filter, before=before, limit=limit):
            yield it

    async def adelete_thread(self, thread_id: str) -> None:
        await self._memory.adelete_thread(thread_id)


def get_checkpointer(
    settings: Settings,
    redis_client: Redis | None = None,
) -> BaseCheckpointSaver:
    """Factory selecting the optimal checkpointer based on configuration and credentials.

    Hierarchy:
    1. If `CHECKPOINTER_BACKEND == "dynamodb"` or (`auto` and AWS credentials exist):
       Use `AsyncDynamoDBSaver` (Serverless, $0 idle cost, auto-expiring TTL).
    2. If `CHECKPOINTER_BACKEND == "redis"` and `redis_client` is available:
       Use `AsyncRedisSaver` (Sub-millisecond memory).
    3. Fallback:
       Use `MemorySaver` (In-memory, optimal for local unit testing and quick dev).
    """
    backend = settings.CHECKPOINTER_BACKEND.lower().strip()

    if backend == "dynamodb" or (backend == "auto" and has_aws_credentials()):
        logger.info(
            "CHECKPOINTER_SELECTED_DYNAMODB",
            table=settings.DYNAMODB_TABLE_SESSIONS,
            ttl_days=settings.CHECKPOINT_TTL_DAYS,
        )
        return AsyncDynamoDBSaver(
            table_name=settings.DYNAMODB_TABLE_SESSIONS,
            ttl_days=settings.CHECKPOINT_TTL_DAYS,
            region_name=settings.AWS_REGION,
        )

    if backend == "redis" and redis_client is not None:
        logger.info("CHECKPOINTER_SELECTED_REDIS", ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS)
        return AsyncRedisSaver(
            redis_client=redis_client,
            ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS,
        )

    logger.info("CHECKPOINTER_SELECTED_MEMORY")
    return MemorySaver()
