from app.services.realtime_transcription import (
    DEFAULT_TRANSCRIBE_MODEL,
    OPENAI_REALTIME_URL,
    _transcription_session_update_event,
)


def test_openai_realtime_url_uses_ga_endpoint() -> None:
    assert OPENAI_REALTIME_URL == "wss://api.openai.com/v1/realtime?intent=transcription"


def test_transcription_session_update_uses_ga_shape() -> None:
    event = _transcription_session_update_event(model=DEFAULT_TRANSCRIBE_MODEL)

    assert event == {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "en",
                    },
                },
            },
        },
    }
