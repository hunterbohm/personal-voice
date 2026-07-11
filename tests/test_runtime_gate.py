import json

import pytest

from personal_voice.runtime_gate import runtime_report_eligible, write_runtime_report

_METRICS = {
    "first_generation_seconds": 1.5,
    "load_seconds": 2.0,
    "peak_rss_mb": 4096.0,
    "warm_mean_seconds": 1.0,
    "warm_p95_seconds": 1.2,
}
_WORKLOAD = {"max_reply_words": 256, "stress_cases": 1}


def test_runtime_report_is_private_machine_and_candidate_bound(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "evaluation/runtime/candidate.json"
    adapter_hash = "a" * 64
    candidate_hash = "c" * 64
    payload = write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
        iterations=20,
        statuses={"rendered": 16, "rejected": 4},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )

    assert payload["ok"] is True
    assert payload["renderer_contract_version"] == 1
    assert report.stat().st_mode & 0o077 == 0
    assert runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
    )
    assert not runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256="d" * 64,
    )

    loaded = json.loads(report.read_text())
    loaded["machine_sha256"] = "f" * 64
    report.write_text(json.dumps(loaded))
    report.chmod(0o600)
    assert not runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
    )

    payload = write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
        iterations=20,
        statuses={"rendered": 20},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    payload["software_versions"]["mlx_lm"] = "different"
    report.write_text(json.dumps(payload))
    report.chmod(0o600)
    assert not runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
    )

    payload = write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
        iterations=20,
        statuses={"rendered": 20},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    payload["renderer_contract_version"] = 2
    report.write_text(json.dumps(payload))
    report.chmod(0o600)
    assert not runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
    )


def test_runtime_report_rejects_underperforming_or_short_benchmarks(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "runtime.json"

    underperforming = write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256="a" * 64,
        candidate_sha256="c" * 64,
        iterations=20,
        statuses={"rendered": 15, "rejected": 5},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    short = write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256="a" * 64,
        candidate_sha256="c" * 64,
        iterations=10,
        statuses={"rendered": 10},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )

    assert underperforming["ok"] is False
    assert short["ok"] is False


def test_runtime_report_enforces_the_18_gib_process_ceiling(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "runtime.json"
    binding = {
        "model_version": "candidate",
        "adapter_sha256": "a" * 64,
        "candidate_sha256": "c" * 64,
    }
    at_ceiling = write_runtime_report(
        report,
        **binding,
        iterations=20,
        statuses={"rendered": 20},
        metrics={**_METRICS, "peak_rss_mb": 18 * 1024},
        workload=_WORKLOAD,
    )

    assert at_ceiling["ok"] is True
    assert runtime_report_eligible(report, **binding)

    above_ceiling = write_runtime_report(
        report,
        **binding,
        iterations=20,
        statuses={"rendered": 20},
        metrics={**_METRICS, "peak_rss_mb": 18 * 1024 + 0.001},
        workload=_WORKLOAD,
    )

    assert above_ceiling["ok"] is False
    assert not runtime_report_eligible(report, **binding)


def test_runtime_report_rejects_negative_status_tampering(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "runtime.json"
    adapter_hash = "a" * 64
    candidate_hash = "c" * 64
    write_runtime_report(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
        iterations=20,
        statuses={"rendered": 20},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    loaded = json.loads(report.read_text())
    loaded["statuses"] = {"rendered": 20, "rejected": -1, "unavailable": 1}
    report.write_text(json.dumps(loaded))
    report.chmod(0o600)

    assert not runtime_report_eligible(
        report,
        model_version="candidate",
        adapter_sha256=adapter_hash,
        candidate_sha256=candidate_hash,
    )


@pytest.mark.parametrize("schema_version", (True, 1.0, 1))
def test_runtime_report_requires_an_integer_schema_version(tmp_path, schema_version) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "runtime.json"
    binding = {
        "model_version": "candidate",
        "adapter_sha256": "a" * 64,
        "candidate_sha256": "c" * 64,
    }
    write_runtime_report(
        report,
        **binding,
        iterations=20,
        statuses={"rendered": 20},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    payload = json.loads(report.read_text())
    payload["schema_version"] = schema_version
    report.write_text(json.dumps(payload))
    report.chmod(0o600)

    assert not runtime_report_eligible(report, **binding)


def test_runtime_report_requires_the_256_word_stress_case(tmp_path) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    report = private / "runtime.json"
    binding = {
        "model_version": "candidate",
        "adapter_sha256": "a" * 64,
        "candidate_sha256": "c" * 64,
    }
    write_runtime_report(
        report,
        **binding,
        iterations=20,
        statuses={"rendered": 20},
        metrics=_METRICS,
        workload=_WORKLOAD,
    )
    payload = json.loads(report.read_text())
    payload["workload"]["stress_cases"] = 0
    report.write_text(json.dumps(payload))
    report.chmod(0o600)

    assert not runtime_report_eligible(report, **binding)
