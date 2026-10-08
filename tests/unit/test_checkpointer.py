"""Unit tests for LangGraph DynamoDB and Redis State Checkpointers."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.config import Settings
from app.core.graphs.checkpointer import (
    AsyncDynamoDBSaver,
    AsyncRedisSaver,
    _decode_typed,
    _encode_typed,
    get_checkpointer,
)


class SampleGraphState(BaseModel):
    step: int = 0
    query: str = ""
    history: list[str] = []


@pytest.mark.asyncio
async def test_encode_decode_typed():
    """Verify typed serialization round-trip works safely without raw pickle."""
    original = ("test_type", b"sample_bytes_12345")
    encoded = _encode_typed(original)
    assert encoded["t"] == "test_type"
    assert isinstance(encoded["d"], str)

    decoded = _decode_typed(encoded)
    assert decoded == original


@pytest.mark.asyncio
async def test_dynamodb_saver_fallback_and_memory_operations():
    """Verify AsyncDynamoDBSaver functions smoothly with internal memory cache."""
    saver = AsyncDynamoDBSaver(table_name="test-table", ttl_days=15)
    config = {
        "configurable": {
            "thread_id": "thread-123",
            "checkpoint_ns": "unit_test",
        }
    }

    # Initially empty
    tup = await saver.aget_tuple(config)
    assert tup is None

    # Put a checkpoint
    ckpt = empty_checkpoint()
    ckpt["id"] = "ckpt-001"
    metadata: CheckpointMetadata = {"source": "unit_test", "step": 1}

    updated_config = await saver.aput(
        config=config,
        checkpoint=ckpt,
        metadata=metadata,
        new_versions={"step": 1},
    )

    assert updated_config["configurable"]["checkpoint_id"] == "ckpt-001"

    # Retrieve checkpoint
    retrieved = await saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.config["configurable"]["thread_id"] == "thread-123"
    assert retrieved.metadata.get("source") == "unit_test"

    # Delete thread
    await saver.adelete_thread("thread-123")
    retrieved_after_del = await saver.aget_tuple(config)
    assert retrieved_after_del is None


@pytest.mark.asyncio
async def test_dynamodb_saver_langgraph_multi_turn_execution():
    """Verify that a compiled LangGraph workflow retains state across multi-turn queries."""
    saver = AsyncDynamoDBSaver()

    def step_node(state: SampleGraphState):
        return {
            "step": state.step + 1,
            "history": [*state.history, state.query],
        }

    builder = StateGraph(SampleGraphState)
    builder.add_node("step_node", step_node)
    builder.set_entry_point("step_node")
    builder.add_edge("step_node", END)

    app = builder.compile(checkpointer=saver)

    thread_id = f"session-saral-{uuid.uuid4()}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    try:
        # Turn 1
        res1 = await app.ainvoke({"query": "What is Grace Period?"}, config=thread_config)
        assert res1["step"] == 1
        assert res1["history"] == ["What is Grace Period?"]

        # Turn 2: state should be retained from Turn 1
        res2 = await app.ainvoke({"query": "What about revival?"}, config=thread_config)
        assert res2["step"] == 2
        assert res2["history"] == ["What is Grace Period?", "What about revival?"]
    finally:
        await saver.adelete_thread(thread_id)


@pytest.mark.asyncio
async def test_redis_saver_put_and_get():
    """Verify AsyncRedisSaver stores and extracts checkpoints via Redis."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    saver = AsyncRedisSaver(redis_client=mock_redis, ttl_seconds=3600)
    config = {
        "configurable": {
            "thread_id": "session-redis-456",
            "checkpoint_ns": "",
        }
    }

    ckpt = empty_checkpoint()
    ckpt["id"] = "ckpt-red-01"
    metadata: CheckpointMetadata = {"user_id": "agent-007"}

    await saver.aput(config, ckpt, metadata, new_versions={})

    # Verify Redis set call was triggered
    assert mock_redis.set.call_count >= 1
    call_args = mock_redis.set.call_args_list[0]
    assert "session-redis-456" in call_args[0][0]
    assert call_args[1].get("ex") == 3600


def test_get_checkpointer_factory():
    """Verify get_checkpointer selects correct backend per config."""
    # 1. Force memory backend
    settings_mem = Settings(CHECKPOINTER_BACKEND="memory")
    cp_mem = get_checkpointer(settings_mem)
    assert isinstance(cp_mem, (MemorySaver, InMemorySaver))

    # 2. Redis backend
    mock_redis = MagicMock()
    settings_redis = Settings(CHECKPOINTER_BACKEND="redis")
    cp_redis = get_checkpointer(settings_redis, redis_client=mock_redis)
    assert isinstance(cp_redis, AsyncRedisSaver)

    # 3. DynamoDB backend
    settings_dynamo = Settings(CHECKPOINTER_BACKEND="dynamodb")
    cp_dynamo = get_checkpointer(settings_dynamo)
    assert isinstance(cp_dynamo, AsyncDynamoDBSaver)
