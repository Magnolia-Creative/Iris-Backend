from app.domains.search import SearchMatch, search_response_dict


def test_search_response_dict_omits_empty_optional_fields():
    payload = search_response_dict(
        matches=[
            SearchMatch(
                clip_id=1,
                local_key="lk",
                start_time_seconds=0.0,
                end_time_seconds=1.0,
                confidence=0.9,
                source="audio",
            )
        ],
        query="car",
    )

    assert payload == {
        "matches": [
            {
                "clip_id": 1,
                "local_key": "lk",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "confidence": 0.9,
                "source": "audio",
            }
        ],
        "query": "car",
    }
