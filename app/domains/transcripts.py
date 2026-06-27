from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranscriptDomainModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CaptionWord(TranscriptDomainModel):
    word: str | None = None
    text: str | None = None
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


class CaptionSentence(TranscriptDomainModel):
    text: str | None = None
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    words: list[CaptionWord] = Field(default_factory=list)


class CaptionMeta(TranscriptDomainModel):
    source_file: str | None = None
    mime_type: str | None = None
    extension: str | None = None
    clip_meta: dict[str, Any] = Field(default_factory=dict)
    video_report: dict[str, Any] = Field(default_factory=dict)


class ClipCaptionsPayload(TranscriptDomainModel):
    project_id: int
    local_key: str
    clip_id: int
    transcript_id: int
    processing_status: str | None = None
    full_text: str = ""
    sentences: list[CaptionSentence] = Field(default_factory=list)
    meta: CaptionMeta = Field(default_factory=CaptionMeta)


def sentences_from_transcript_payload(transcript_payload: dict[str, Any]) -> list[CaptionSentence]:
    raw = transcript_payload.get("segments")
    if not isinstance(raw, list):
        return []
    sentences: list[CaptionSentence] = []
    for segment in raw:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        sentences.append(
            CaptionSentence(
                text=segment.get("text") if isinstance(segment.get("text"), str) else None,
                start=segment.get("start") if isinstance(segment.get("start"), int | float) else None,
                end=segment.get("end") if isinstance(segment.get("end"), int | float) else None,
                confidence=segment.get("confidence")
                if isinstance(segment.get("confidence"), int | float)
                else None,
                words=[
                    CaptionWord.model_validate(word)
                    for word in words
                    if isinstance(word, dict)
                ]
                if isinstance(words, list)
                else [],
            )
        )
    return sentences


def captions_payload_from_transcript(
    *,
    project_id: int,
    local_key: str,
    clip_id: int,
    transcript_id: int,
    processing_status: str | None,
    transcript_payload: dict[str, Any],
) -> ClipCaptionsPayload:
    full_text = transcript_payload.get("full_text")
    clip_meta = transcript_payload.get("clip_meta")
    video_report = transcript_payload.get("video_report")
    return ClipCaptionsPayload(
        project_id=project_id,
        local_key=local_key,
        clip_id=clip_id,
        transcript_id=transcript_id,
        processing_status=processing_status,
        full_text=full_text if isinstance(full_text, str) else "",
        sentences=sentences_from_transcript_payload(transcript_payload),
        meta=CaptionMeta(
            source_file=transcript_payload.get("source_file"),
            mime_type=transcript_payload.get("mime_type"),
            extension=transcript_payload.get("extension"),
            clip_meta=clip_meta if isinstance(clip_meta, dict) else {},
            video_report=video_report if isinstance(video_report, dict) else {},
        ),
    )
