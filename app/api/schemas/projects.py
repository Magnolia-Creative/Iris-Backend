from pydantic import BaseModel


class AgentSessionCreatePayload(BaseModel):
    project_name: str | None = None
    session_name: str | None = None


class ProjectCreatePayload(BaseModel):
    name: str | None = None


class AgentSessionForProjectCreatePayload(BaseModel):
    session_name: str | None = None
