from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    clips: Mapped[list["Clip"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class Clip(Base):
    __tablename__ = "clips"
    __table_args__ = (Index("idx_clips_project_id", "project_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    codec: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_rate: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    duration_seconds: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    language_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="clips")
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="clip", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    summary: Mapped["Summary | None"] = relationship(
        back_populates="clip", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )


class Transcript(Base):
    __tablename__ = "transcripts"
    __table_args__ = (
        Index("idx_transcripts_clip_id", "clip_id"),
        Index("idx_transcripts_jsonb_gin", "transcript", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    transcript: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    clip: Mapped["Clip"] = relationship(back_populates="transcript")


class Summary(Base):
    __tablename__ = "summaries"
    __table_args__ = (Index("idx_summaries_clip_id", "clip_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    clip: Mapped["Clip"] = relationship(back_populates="summary")
