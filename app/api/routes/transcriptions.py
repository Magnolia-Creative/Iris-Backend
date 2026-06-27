from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import ClerkPrincipal, require_clerk_user
from app.database import get_db
from app.services.transcription import print_received_transcript, transcribe_upload_to_sentences
from app.services.transcript_store import insert_sentence_upload_transcript


router = APIRouter()


@router.post("/transcriptions/sentences")
async def create_sentence_transcription(
    audio: UploadFile = File(...),
    _: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    suffix = ""
    if audio.filename and "." in audio.filename:
        suffix = f".{audio.filename.rsplit('.', 1)[-1]}"

    result = await transcribe_upload_to_sentences(audio_bytes, suffix=suffix or ".m4a")
    provider_transcript_id = result.get("transcript_id")
    db_transcript_id = await insert_sentence_upload_transcript(db, result)
    result["transcript_id"] = db_transcript_id
    meta = result.get("meta")
    meta_out = dict(meta) if isinstance(meta, dict) else {}
    meta_out["provider_transcript_id"] = provider_transcript_id
    result["meta"] = meta_out
    print_received_transcript(
        source="main_endpoint",
        transcript_id=db_transcript_id,
        full_text=result.get("full_text"),
        segments=result.get("sentences"),
    )
    return result
