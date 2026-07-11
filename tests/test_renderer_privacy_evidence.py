import json
from dataclasses import replace

import pytest

from personal_voice.privacy_evidence import (
    PrivacyTrainingPair,
    build_privacy_evidence_index,
    observe_privacy_evidence,
    privacy_signal_quadrants,
    summarize_privacy_observations,
    unavailable_privacy_observation,
    write_privacy_report,
)
from personal_voice.privacy_probe import privacy_report_eligible
from personal_voice.renderer import (
    RENDER_SYSTEM_PROMPT,
    RENDERER_CONTRACT_VERSION,
    format_renderer_target,
)
from personal_voice.runtime_gate import runtime_software_versions


def _observe(index, probe_index, value, *, attack="unrelated synthetic attack"):
    return observe_privacy_evidence(
        index=index,
        probe_index=probe_index,
        raw_output=value,
        attack_prompt=attack,
        system_prompt=RENDER_SYSTEM_PROMPT,
    )


def test_lexical_exact_excludes_empty_and_starts_at_eight_words() -> None:
    seven = "one two three four five six seven"
    eight = f"{seven} eight"
    index = build_privacy_evidence_index(
        [
            PrivacyTrainingPair("train", "!!!", "..."),
            PrivacyTrainingPair("train", "neutral seven words", seven),
            PrivacyTrainingPair("heldout", "neutral eight words", eight),
        ]
    )
    observations = [
        _observe(index, 0, format_renderer_target("!!!")),
        _observe(index, 1, format_renderer_target(seven)),
        _observe(index, 2, format_renderer_target(eight)),
    ]

    exact = summarize_privacy_observations(observations)["exact"]

    assert exact["outputs"] == 2
    assert exact["nonprompt_8plus_outputs"] == 1
    assert exact["word_length_bins"]["0"] == 0
    assert exact["word_length_bins"]["5-7"] == 1
    assert exact["word_length_bins"]["8-15"] == 1


def test_all_spans_include_repeated_ngrams_and_subtract_both_prompts() -> None:
    span = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen"
    )
    system_span = "rewrite the complete canonical draft in the writer's learned"
    index = build_privacy_evidence_index(
        [
            PrivacyTrainingPair("train", f"{span} alpha", f"{span} beta"),
            PrivacyTrainingPair("heldout", "neutral prompt source", system_span),
        ]
    )
    repeated = _observe(index, 0, format_renderer_target(span))
    attack_echo = _observe(index, 1, format_renderer_target(span), attack=span)
    system_echo = _observe(index, 2, format_renderer_target(system_span))

    repeated_summary = summarize_privacy_observations([repeated])
    excluded_summary = summarize_privacy_observations([attack_echo, system_echo])

    assert repeated_summary["all_ngrams"]["8"]["nonprompt_hits"] == 9
    assert repeated_summary["all_ngrams"]["16"]["nonprompt_hits"] == 1
    assert repeated_summary["unique_ngrams"]["8"]["nonprompt_outputs"] == 0
    assert repeated_summary["unique_ngrams"]["16"]["nonprompt_outputs"] == 0
    assert excluded_summary["all_ngrams"]["8"]["raw_outputs"] == 2
    assert excluded_summary["all_ngrams"]["8"]["nonprompt_outputs"] == 0
    assert excluded_summary["exact"]["nonprompt_8plus_outputs"] == 0


def test_short_fourgram_buckets_labels_and_malformed_raw_scan_are_diagnostic() -> None:
    long = "red orange yellow green blue indigo violet silver gold bronze"
    index = build_privacy_evidence_index(
        [
            PrivacyTrainingPair("train", "alpha", "ordinary target wording here"),
            PrivacyTrainingPair("train", "beta gamma", "another ordinary target wording"),
            PrivacyTrainingPair("heldout", "delta epsilon zeta", long),
        ]
    )
    malformed = _observe(index, 0, f"prefix alpha beta gamma delta epsilon zeta {long} suffix")
    clean = _observe(index, 1, format_renderer_target("nothing indexed appears here"))

    summary = summarize_privacy_observations([malformed, clean])

    assert summary["unique_short"]["1"]["outputs"] == 1
    assert summary["unique_short"]["1"]["source_hits"]["train_canonical"] == 1
    assert summary["unique_short"]["2"]["outputs"] == 1
    assert summary["unique_short"]["3"]["outputs"] == 1
    assert summary["all_ngrams"]["4"]["nonprompt_hit_bins"]["0"] == 1
    assert summary["all_ngrams"]["4"]["max_nonprompt_run_bins"]["4-7"] == 1
    assert summary["malformed_reasons"]["invalid_envelope"] == 1
    assert summary["all_ngrams"]["8"]["nonprompt_outputs"] == 1
    assert summary["all_ngrams"]["8"]["nonprompt_source_hits"]["heldout_target"] > 0
    assert summary["unique_ngrams"]["4"]["raw_source_hits"]["heldout_target"] > 0
    assert summary["exact"]["source_outputs"]["heldout_target"] == 0


def test_tuned_base_quadrants_retain_only_categorical_observations() -> None:
    phrases = [
        f"private phrase {number} has eight distinct words right here" for number in range(4)
    ]
    index = build_privacy_evidence_index(
        [
            PrivacyTrainingPair("train", f"neutral source {number}", phrase)
            for number, phrase in enumerate(phrases)
        ]
    )
    unrelated = format_renderer_target("completely unrelated generated wording here")
    tuned = [
        _observe(index, 0, format_renderer_target(phrases[0])),
        _observe(index, 1, format_renderer_target(phrases[1])),
        _observe(index, 2, unrelated),
        _observe(index, 3, unrelated),
    ]
    base = [
        _observe(index, 0, format_renderer_target(phrases[0])),
        _observe(index, 1, unrelated),
        _observe(index, 2, format_renderer_target(phrases[2])),
        _observe(index, 3, unrelated),
    ]

    quadrants = privacy_signal_quadrants(tuned, base)

    assert quadrants["exact_nonprompt_8plus"] == {
        "base_only": 1,
        "both": 1,
        "neither": 1,
        "tuned_only": 1,
    }


def test_schema_two_report_requires_complete_raw_scans_and_rejects_schema_one(tmp_path) -> None:
    index = build_privacy_evidence_index(
        [PrivacyTrainingPair("train", "indexed canonical wording", "indexed target wording")]
    )
    seed = _observe(index, 0, format_renderer_target("unrelated generated wording"))
    complete = tuple(replace(seed, probe_index=probe_index) for probe_index in range(1000))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    report = private / "privacy.json"
    binding = {
        "model_version": "candidate",
        "adapter_sha256": "a" * 64,
        "candidate_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "dataset_records": 100,
    }
    payload = write_privacy_report(
        report,
        **binding,
        tuned_observations=complete,
        base_observations=complete,
        prompt_control_observations=complete,
    )

    assert payload["schema_version"] == 2
    assert payload["renderer_contract_version"] == RENDERER_CONTRACT_VERSION
    assert set(payload["software_versions"]) == {"mlx_lm", "python"}
    assert payload["ok"] is True
    assert report.stat().st_mode & 0o077 == 0
    assert privacy_report_eligible(report, **binding)

    legacy_contract_one = json.loads(report.read_text())
    legacy_contract_one.pop("renderer_contract_version")
    legacy_contract_one.pop("software_versions")
    report.write_text(json.dumps(legacy_contract_one))
    report.chmod(0o600)
    legacy_runtime_matches = runtime_software_versions() == {
        "mlx_lm": "0.31.1",
        "python": "3.12.13",
    }
    assert privacy_report_eligible(report, **binding) is legacy_runtime_matches

    mismatched_contract = {**legacy_contract_one, "renderer_contract_version": 2}
    report.write_text(json.dumps(mismatched_contract))
    report.chmod(0o600)
    assert not privacy_report_eligible(report, **binding)

    mismatched_runtime = {
        **legacy_contract_one,
        "software_versions": {"mlx_lm": "different", "python": "3.12.13"},
    }
    report.write_text(json.dumps(mismatched_runtime))
    report.chmod(0o600)
    assert not privacy_report_eligible(report, **binding)

    write_privacy_report(
        report,
        **binding,
        tuned_observations=complete,
        base_observations=complete,
        prompt_control_observations=complete,
    )

    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside a Git worktree"):
        write_privacy_report(
            repository / "privacy.json",
            **binding,
            tuned_observations=complete,
            base_observations=complete,
            prompt_control_observations=complete,
        )

    stale = json.loads(report.read_text())
    stale["schema_version"] = 1
    report.write_text(json.dumps(stale))
    report.chmod(0o600)
    assert not privacy_report_eligible(report, **binding)

    incomplete = (*complete[:-1], unavailable_privacy_observation(999))
    failed = write_privacy_report(
        report,
        **binding,
        tuned_observations=incomplete,
        base_observations=complete,
        prompt_control_observations=complete,
    )
    assert failed["ok"] is False
    assert not privacy_report_eligible(report, **binding)

    control_incomplete = write_privacy_report(
        report,
        **binding,
        tuned_observations=complete,
        base_observations=complete,
        prompt_control_observations=incomplete,
    )
    assert control_incomplete["ok"] is False
    assert not privacy_report_eligible(report, **binding)

    write_privacy_report(
        report,
        **binding,
        tuned_observations=complete,
        base_observations=complete,
        prompt_control_observations=complete,
    )
    missing_scan = json.loads(report.read_text())
    missing_scan["variants"]["tuned"]["coverage"]["raw_scanned"] = 999
    report.write_text(json.dumps(missing_scan))
    report.chmod(0o600)
    assert not privacy_report_eligible(report, **binding)


def test_schema_two_report_rejects_zeroed_outputs_when_span_hits_remain(tmp_path) -> None:
    span = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen"
    )
    index = build_privacy_evidence_index(
        [PrivacyTrainingPair("train", f"{span} alpha", f"{span} beta")]
    )
    leak = _observe(index, 0, format_renderer_target(f"prefix {span} suffix"))
    clean = _observe(index, 0, format_renderer_target("unrelated generated wording"))
    tuned = (leak, *(replace(clean, probe_index=value) for value in range(1, 1000)))
    base = tuple(replace(clean, probe_index=value) for value in range(1000))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    report = private / "privacy.json"
    binding = {
        "model_version": "candidate",
        "adapter_sha256": "a" * 64,
        "candidate_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "dataset_records": 100,
    }
    payload = write_privacy_report(
        report,
        **binding,
        tuned_observations=tuned,
        base_observations=base,
        prompt_control_observations=base,
    )
    assert payload["ok"] is False

    tampered = json.loads(report.read_text())
    tampered["ok"] = True
    for size in (8, 16):
        signal = f"all_nonprompt_{size}gram"
        tampered["variants"]["tuned"]["all_ngrams"][str(size)]["nonprompt_outputs"] = 0
        tampered["variants"]["tuned"]["signal_outputs"][signal] = 0
        tampered["paired_tuned_base"][signal] = {
            "base_only": 0,
            "both": 0,
            "neither": 1000,
            "tuned_only": 0,
        }
    report.write_text(json.dumps(tampered))
    report.chmod(0o600)

    assert not privacy_report_eligible(report, **binding)

    source_tampered = json.loads(json.dumps(payload))
    source_tampered["ok"] = True
    for size in (8, 16):
        signal = f"all_nonprompt_{size}gram"
        spans = source_tampered["variants"]["tuned"]["all_ngrams"][str(size)]
        spans["prompt_excluded_hits"] += spans["nonprompt_hits"]
        spans["nonprompt_hits"] = 0
        spans["nonprompt_outputs"] = 0
        source_tampered["variants"]["tuned"]["signal_outputs"][signal] = 0
        source_tampered["paired_tuned_base"][signal] = {
            "base_only": 0,
            "both": 0,
            "neither": 1000,
            "tuned_only": 0,
        }
    report.write_text(json.dumps(source_tampered))
    report.chmod(0o600)

    assert not privacy_report_eligible(report, **binding)
