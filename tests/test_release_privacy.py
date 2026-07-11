from personal_voice.release_privacy import scan_paths


def test_release_privacy_scan_reports_categories_without_content(tmp_path) -> None:
    safe = tmp_path / "safe.txt"
    safe.write_text("Contact writer@example.com or +1 (555) 010-0200.")

    private_email = tmp_path / "email.txt"
    private_email.write_text("person" + "@mail.invalid")

    absolute_path = tmp_path / "path.txt"
    absolute_path.write_text("/" + "Users" + "/someone/private.txt")

    key = tmp_path / "key.txt"
    key.write_text("AKIA" + "A" * 16)

    database = tmp_path / "corpus.sqlite"
    database.write_text("synthetic")

    issues = scan_paths(tmp_path, [safe, private_email, absolute_path, key, database])

    assert issues == [
        ("corpus.sqlite", "private-artifact-extension"),
        ("email.txt", "non-example-email"),
        ("key.txt", "aws-access-key"),
        ("path.txt", "absolute-home-path"),
    ]
