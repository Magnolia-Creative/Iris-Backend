from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.agent.intent.models import IntentAgentRequest


class IntentAgentRunCreatePayload(IntentAgentRequest):
    kind: Literal["intent"]


class AutomakeAgentRunCreatePayload(BaseModel):
    kind: Literal["automake"]
    project_id: int | None = None
    project_name: str | None = None
    session_name: str | None = None


AgentRunCreatePayload = Annotated[
    IntentAgentRunCreatePayload | AutomakeAgentRunCreatePayload,
    Field(discriminator="kind"),
]
