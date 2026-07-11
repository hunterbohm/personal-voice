import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from personal_voice.privacy_evidence import PrivacyTrainingPair
from personal_voice.renderer import format_renderer_target

_SPEC = spec_from_file_location(
    "renderer_privacy_probe",
    Path(__file__).resolve().parents[1] / "scripts/renderer_privacy_probe.py",
)
assert _SPEC is not None and _SPEC.loader is not None
renderer_privacy_probe = module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer_privacy_probe)


def test_script_runs_same_full_matrix_for_tuned_base_and_prints_no_hashes_or_text(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls = {"base": [], "tuned": []}

    class Backend:
        version = "candidate"
        artifact_hash = "a" * 64
        candidate_hash = "c" * 64

        def __init__(self, variant):
            self.variant = variant

        def generate(self, medium, _attack):
            calls[self.variant].append(medium)
            return format_renderer_target("synthetic generated wording")

    monkeypatch.setattr(
        renderer_privacy_probe,
        "load_mlx_inprocess_backend",
        lambda _manifest, use_adapter=True: Backend("tuned" if use_adapter else "base"),
    )
    monkeypatch.setattr(
        renderer_privacy_probe,
        "require_candidate_dataset_binding",
        lambda _manifest, _data: ("d" * 64, 3),
    )
    monkeypatch.setattr(
        renderer_privacy_probe,
        "_load_pairs",
        lambda _data: [
            PrivacyTrainingPair("train", "synthetic canonical wording", "synthetic target wording")
        ],
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    report = private / "privacy.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "renderer_privacy_probe.py",
            "--manifest",
            str(private / "candidate.json"),
            "--data",
            str(private / "data"),
            "--report",
            str(report),
        ],
    )

    assert renderer_privacy_probe.main() == 0

    stdout = capsys.readouterr().out
    public = json.loads(stdout)
    assert len(calls["base"]) == len(calls["tuned"]) == 1000
    assert calls["base"] == calls["tuned"]
    assert calls["tuned"][:4] == ["text", "dm", "email", "public"]
    assert public["schema_version"] == 2
    assert public["activation_eligible"] is True
    assert set(public) == {
        "activation_eligible",
        "dataset_records",
        "evidence_kind",
        "model_version",
        "ok",
        "paired_tuned_base",
        "probes",
        "renderer_contract_version",
        "report_path",
        "schema_version",
        "software_versions",
        "variants",
    }
    assert "adapter_sha256" not in stdout
    assert "candidate_sha256" not in stdout
    assert "dataset_sha256" not in stdout
    assert "a" * 64 not in stdout
    assert "c" * 64 not in stdout
    assert "d" * 64 not in stdout
    assert "synthetic generated wording" not in stdout
