from typing import Literal

from pydantic import BaseModel

from app.agent.intent.editing.models import IntentCompilerContext


class VoiceIntentStartPayload(BaseModel):
    type: Literal["start"]
    context: IntentCompilerContext
