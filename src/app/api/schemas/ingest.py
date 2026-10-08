"""Schemas for policy document ingestion and vector collection management."""

from datetime import date

from pydantic import BaseModel, Field

from app.api.schemas.common import DocumentType, ProductLine


class IngestDocumentMetadata(BaseModel):
    """Metadata for an ingested insurance policy document."""

    document_title: str = Field(..., min_length=2, max_length=255)
    document_type: DocumentType = DocumentType.MASTER_POLICY
    product_name: str = Field(..., min_length=2, max_length=100)
    product_code: str = Field(..., min_length=2, max_length=50)
    uin: str | None = Field(
        default=None, max_length=50, description="IRDAI Unique Identification Number"
    )
    product_line: ProductLine = ProductLine.TERM
    state: str = Field(default="ALL", max_length=5)
    effective_date: str = Field(default_factory=lambda: date.today().isoformat())
    expiration_date: str | None = None
    is_state_specific: bool = False


class IngestResponse(BaseModel):
    """Result of policy document processing and vector ingestion."""

    document_id: str
    document_title: str
    chunks_created: int
    status: str = "completed"
    message: str
    collection_name: str
    s3_uri: str | None = None
    download_url: str | None = None


class CollectionInfo(BaseModel):
    """Vector database collection statistics."""

    collection_name: str
    points_count: int = 0
    vectors_count: int = 0
    vector_dimension: int
    status: str = "green"
