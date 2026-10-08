"""Policy query and question-answering endpoints for Indian Life Insurance.

Follows SOLID Single Responsibility Principle: this controller is solely
responsible for HTTP routing and response dispatch, delegating domain
orchestration to QueryService.
"""

from fastapi import APIRouter

from app.api.deps import CurrentUser, QueryServiceDep
from app.api.schemas.query import PolicyAnswerResponse, PolicyQueryRequest

router = APIRouter(tags=["Policy Query"])


@router.post(
    "/query",
    response_model=PolicyAnswerResponse,
    summary="Query Indian Life Insurance Policies (Agentic RAG)",
    description="Submit a natural language question about policy terms, clauses, riders, exclusions, or IRDAI regulations.",
)
async def query_policy(
    request: PolicyQueryRequest,
    current_user: CurrentUser,
    query_service: QueryServiceDep,
) -> PolicyAnswerResponse:
    """Execute policy query against the LangGraph Agentic RAG pipeline via QueryService."""
    return await query_service.execute_query(request, current_user)
