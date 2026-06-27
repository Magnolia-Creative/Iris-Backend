from typing import Literal

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int | None = None


class SourceSearchRequest(SearchRequest):
    mode: Literal["semantic", "transcript"]
