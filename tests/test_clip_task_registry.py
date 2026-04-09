import asyncio

from app.services.clip_task_registry import ClipTaskRegistry


def test_clip_task_registry_registers_cancels_and_clears_tasks():
    async def exercise_registry():
        registry = ClipTaskRegistry()

        blocker = asyncio.Event()
        task = asyncio.create_task(blocker.wait())
        await registry.register(session_id=7, local_key="abc-123", task=task)

        assert await registry.active_count_for_session(7) == 1
        assert await registry.cancel(session_id=7, local_key="abc-123") is True

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert await registry.active_count_for_session(7) == 0
        assert await registry.cancel(session_id=7, local_key="abc-123") is False

    asyncio.run(exercise_registry())
