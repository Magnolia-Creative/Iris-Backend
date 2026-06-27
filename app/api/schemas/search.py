from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int | None = None
