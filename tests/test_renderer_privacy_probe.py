import json

import pytest

from personal_voice.privacy_probe import (
    MAX_PRIVACY_PROBES,
    build_privacy_target_index,
    examine_privacy_output,
    normalize_privacy_text,
    privacy_attack,
    privacy_output_matches,
    require_candidate_dataset_binding,
)
from personal_voice.renderer import format_renderer_target, literal_counter
from personal_voice.training_data import renderer_dataset_identity


def test_privacy_attacks_are_unique_and_contain_no_guarded_literals() -> None:
    attacks = [privacy_attack(index) for index in range(MAX_PRIVACY_PROBES)]

    assert MAX_PRIVACY_PROBES == 1000
    assert len(set(attacks)) == MAX_PRIVACY_PROBES
    assert all(not literal_counter(attack) for attack in attacks)


def test_privacy_normalization_aligns_slotized_and_restored_short_targets() -> None:
    slotized = "send the private draft at [[PV_LITERAL_0]] please"
    restored = "send the private draft at 3:30 please"

    assert normalize_privacy_text(slotized) == normalize_privacy_text(restored)


def test_privacy_output_scan_catches_short_and_embedded_memorized_targets() -> None:
    short_index = build_privacy_target_index(["private alpha beta"])
    status, scanned = examine_privacy_output("prefix private alpha beta suffix")
    exact, short, _ngrams = privacy_output_matches(scanned, short_index)

    assert status == "malformed"
    assert exact == 0
    assert short == 1

    longer = "private alpha beta gamma delta epsilon zeta"
    long_index = build_privacy_target_index([longer])
    status, scanned = examine_privacy_output(format_renderer_target(f"{longer} suffix"))
    _exact, _short, ngrams = privacy_output_matches(scanned, long_index)

    assert status == "examined"
    assert ngrams[4] == 1


def test_candidate_dataset_binding_rejects_changed_private_splits(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    data = private / "data"
    data.mkdir(mode=0o700)
    for split in ("train", "valid", "test"):
        path = data / f"{split}.jsonl"
        path.write_text(f"{split}\n")
        path.chmod(0o600)
    dataset_hash, dataset_records = renderer_dataset_identity(data)
    manifest = private / "candidate.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training_dataset_records": dataset_records,
                "training_dataset_sha256": dataset_hash,
            }
        )
    )
    manifest.chmod(0o600)

    assert require_candidate_dataset_binding(manifest, data) == (
        dataset_hash,
        dataset_records,
    )
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 2.0
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o600)
    with pytest.raises(ValueError, match="binding is invalid"):
        require_candidate_dataset_binding(manifest, data)
    payload["schema_version"] = 2
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o600)
    changed = data / "test.jsonl"
    changed.write_text("changed\n")
    changed.chmod(0o600)
    with pytest.raises(ValueError, match="binding is invalid"):
        require_candidate_dataset_binding(manifest, data)
