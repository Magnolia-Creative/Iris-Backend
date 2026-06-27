from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import ClerkPrincipal, require_clerk_user
from app.database import get_db


router = APIRouter()


@router.get("/")
def health():
    return {"status": "ok"}


@router.get("/db-health")
async def db_health(
    _: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("SELECT 1"))
    return {"database": "ok", "result": result.scalar_one()}
