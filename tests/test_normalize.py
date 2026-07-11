from personal_voice.normalize import is_usable_text, normalize_text


def test_normalize_text_redacts_international_phones_and_preserves_url_punctuation() -> None:
    text = "call +1 (555) 010-0200, then see https://example.com/path?q=1."

    assert normalize_text(text) == "call <|phone|>, then see <|url|>."


def test_is_usable_text_rejects_identifier_only_rows_but_keeps_emoji() -> None:
    assert not is_usable_text("<|url|>.")
    assert not is_usable_text("<|phone|>")
    assert is_usable_text("😂")
