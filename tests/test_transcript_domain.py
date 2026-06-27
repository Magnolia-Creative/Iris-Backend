from app.domains.transcripts import captions_payload_from_transcript, sentences_from_transcript_payload


def test_sentences_from_transcript_payload_normalizes_segments():
    sentences = sentences_from_transcript_payload(
        {
            "segments": [
                {
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.5,
                    "confidence": 0.9,
                    "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
                },
                "bad",
            ]
        }
    )

    assert len(sentences) == 1
    assert sentences[0].text == "Hello"
    assert sentences[0].words[0].word == "Hello"


def test_captions_payload_from_transcript_defaults_unstructured_meta():
    payload = captions_payload_from_transcript(
        project_id=1,
        local_key="lk",
        clip_id=2,
        transcript_id=3,
        processing_status="ready",
        transcript_payload={
            "full_text": "Hello",
            "segments": [{"text": "Hello"}],
            "clip_meta": "bad",
            "video_report": None,
        },
    )

    dumped = payload.model_dump(exclude_none=True)
    assert dumped["project_id"] == 1
    assert dumped["full_text"] == "Hello"
    assert dumped["sentences"][0]["text"] == "Hello"
    assert dumped["meta"]["clip_meta"] == {}
    assert dumped["meta"]["video_report"] == {}
