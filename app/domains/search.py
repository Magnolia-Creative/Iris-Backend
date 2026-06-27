from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchDomainModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchMatch(SearchDomainModel):
    clip_id: int
    local_key: str
    file_name: str | None = None
    start_time_seconds: float
    end_time_seconds: float
    confidence: float
    source: str
    match_text: str | None = None


class SearchResponse(SearchDomainModel):
    matches: list[SearchMatch] = Field(default_factory=list)
    query: str
    disabled_reason: str | None = None
    error: str | None = None


def search_response_dict(
    *,
    matches: list[SearchMatch | dict[str, Any]],
    query: str,
    disabled_reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    response = SearchResponse(
        matches=[SearchMatch.model_validate(match) for match in matches],
        query=query,
        disabled_reason=disabled_reason,
        error=error,
    )
    return response.model_dump(exclude_none=True)
