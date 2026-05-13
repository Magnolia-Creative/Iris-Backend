from app.services.sentence_transcript_payload import intent_jsonb_from_sentence_api_result


def test_intent_jsonb_sentence_level_coarse_word():
    j = intent_jsonb_from_sentence_api_result(
        {
            "full_text": "Hello world.",
            "sentences": [{"text": "Hello world.", "start": 0.0, "end": 1.2}],
        }
    )
    assert j["full_text"] == "Hello world."
    assert len(j["segments"]) == 1
    assert j["segments"][0]["words"][0]["word"] == "Hello world."
    assert j["segments"][0]["words"][0]["start"] == 0.0
    assert j["segments"][0]["words"][0]["end"] == 1.2


def test_intent_jsonb_uses_nested_words_when_present():
    j = intent_jsonb_from_sentence_api_result(
        {
            "full_text": "Hi there",
            "sentences": [
                {
                    "text": "Hi there",
                    "words": [
                        {"word": "Hi", "start": 0.0, "end": 0.1},
                        {"word": "there", "start": 0.15, "end": 0.4},
                    ],
                }
            ],
        }
    )
    assert len(j["segments"]) == 1
    assert len(j["segments"][0]["words"]) == 2
    assert j["segments"][0]["words"][0]["word"] == "Hi"
