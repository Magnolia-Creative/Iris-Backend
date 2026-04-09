import asyncio
from dataclasses import dataclass


@dataclass
class RegisteredClipTask:
    session_id: int
    local_key: str
    task: asyncio.Task


class ClipTaskRegistry:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[tuple[int, str], RegisteredClipTask] = {}

    async def register(self, *, session_id: int, local_key: str, task: asyncio.Task) -> None:
        key = (session_id, local_key)
        async with self._lock:
            self._tasks[key] = RegisteredClipTask(session_id=session_id, local_key=local_key, task=task)

    async def pop(self, *, session_id: int, local_key: str) -> RegisteredClipTask | None:
        key = (session_id, local_key)
        async with self._lock:
            return self._tasks.pop(key, None)

    async def cancel(self, *, session_id: int, local_key: str) -> bool:
        registered = await self.pop(session_id=session_id, local_key=local_key)
        if registered is None:
            return False
        registered.task.cancel()
        return True

    async def active_count_for_session(self, session_id: int) -> int:
        async with self._lock:
            return sum(1 for key in self._tasks if key[0] == session_id)


clip_task_registry = ClipTaskRegistry()
