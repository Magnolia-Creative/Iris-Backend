from typing import Literal

from pydantic import BaseModel


class WebSocketSessionStartPayload(BaseModel):
    type: Literal["start_session"]
    user_prompt: str


class WebSocketRepromptPayload(BaseModel):
    type: Literal["reprompt"]
    prompt: str
