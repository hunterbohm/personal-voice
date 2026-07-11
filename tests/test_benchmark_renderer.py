import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location(
    "benchmark_renderer",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_renderer.py",
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_renderer = module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_renderer)


def test_runtime_benchmark_exercises_the_supported_reply_envelope(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    word_counts: list[int] = []

    class SyntheticBackend:
        version = "candidate"
        artifact_hash = "a" * 64
        candidate_hash = "c" * 64

    def render(draft, _medium, _backend):
        word_counts.append(len(draft.split()))
        return {"status": "rendered"}

    monkeypatch.setattr(
        benchmark_renderer,
        "load_mlx_inprocess_backend",
        lambda _manifest: SyntheticBackend(),
    )
    monkeypatch.setattr(benchmark_renderer, "render_draft", render)
    monkeypatch.setattr(
        benchmark_renderer,
        "write_runtime_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        benchmark_renderer,
        "runtime_report_eligible",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_renderer.py",
            "--manifest",
            str(tmp_path / "candidate.json"),
            "--iterations",
            "20",
            "--report",
            str(tmp_path / "runtime.json"),
        ],
    )

    assert benchmark_renderer.main() == 0

    report = json.loads(capsys.readouterr().out)
    assert len(word_counts) == 20
    assert max(word_counts) == 256
    assert word_counts.count(256) == 1
    assert report["iterations"] == 20
    assert "draft" not in report
