import asyncio

from app.services.clip_task_registry import ClipTaskRegistry


def test_clip_task_registry_registers_cancels_and_clears_tasks():
    async def exercise_registry():
        registry = ClipTaskRegistry()

        blocker = asyncio.Event()
        task = asyncio.create_task(blocker.wait())
        await registry.register(project_id=11, local_key="abc-123", task=task)

        assert await registry.active_count_for_project(11) == 1
        assert await registry.cancel(project_id=11, local_key="abc-123") is True

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert await registry.active_count_for_project(11) == 0
        assert await registry.cancel(project_id=11, local_key="abc-123") is False

    asyncio.run(exercise_registry())
