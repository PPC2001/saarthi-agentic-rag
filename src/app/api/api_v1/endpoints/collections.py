"""Vector collection management endpoints."""

from fastapi import APIRouter, HTTPException, status
from qdrant_client.http.models import Distance, VectorParams

from app.api.deps import CurrentUser, QdrantDep, SettingsDep
from app.api.schemas.ingest import CollectionInfo
from app.observability.logging import get_logger

router = APIRouter(tags=["Collections"])
logger = get_logger("collections_endpoint")


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections(
    qdrant: QdrantDep,
    current_user: CurrentUser,
) -> list[CollectionInfo]:
    """List available vector database collections and statistics."""
    try:
        response = await qdrant.get_collections()
        results: list[CollectionInfo] = []
        for col in response.collections:
            info = await qdrant.get_collection(col.name)
            vector_size = 0
            if info.config.params.vectors and hasattr(info.config.params.vectors, "size"):
                vector_size = info.config.params.vectors.size

            results.append(
                CollectionInfo(
                    collection_name=col.name,
                    points_count=info.points_count or 0,
                    vectors_count=info.indexed_vectors_count or 0,
                    vector_dimension=vector_size,
                    status=str(info.status.value)
                    if hasattr(info.status, "value")
                    else str(info.status),
                )
            )
        logger.info(
            "COLLECTIONS_LIST_RETRIEVED", count=len(results), user_id=current_user.user_id
        )
        return results
    except Exception as e:
        logger.error("Failed to list collections", error=str(e))
        return []


@router.post("/collections/init", response_model=CollectionInfo)
async def init_collection(
    qdrant: QdrantDep,
    settings: SettingsDep,
    current_user: CurrentUser,
) -> CollectionInfo:
    """Initialize the insurance policies Qdrant collection with configured dimensions."""
    col_name = settings.QDRANT_COLLECTION_NAME
    dimensions = settings.EMBEDDING_DIMENSIONS

    try:
        # Check if collection exists
        collections = await qdrant.get_collections()
        exists = any(c.name == col_name for c in collections.collections)

        if not exists:
            logger.info(
                "Creating Qdrant collection",
                collection_name=col_name,
                dimension=dimensions,
            )
            await qdrant.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
            )

        info = await qdrant.get_collection(col_name)
        logger.info(
            "COLLECTION_INITIALIZED_SUCCESS",
            collection_name=col_name,
            points_count=info.points_count or 0,
        )
        return CollectionInfo(
            collection_name=col_name,
            points_count=info.points_count or 0,
            vectors_count=info.indexed_vectors_count or 0,
            vector_dimension=dimensions,
            status=str(info.status.value)
            if hasattr(info.status, "value")
            else str(info.status),
        )
    except Exception as e:
        logger.error("Failed to initialize collection", error=str(e), collection_name=col_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        ) from e
