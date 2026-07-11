"""Candidate-bound evidence for the live local renderer benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from personal_voice.local_only import inside_git_worktree, write_private_json
from personal_voice.renderer import RENDERER_CONTRACT_VERSION

RUNTIME_STATUS_KEYS = ("rendered", "rejected", "unavailable")
MIN_RUNTIME_ITERATIONS = 20
MIN_RUNTIME_RENDER_RATE = 0.8
# Conservative default for a 24 GiB host. The benchmark reports the actual RSS so
# operators can choose a stricter deployment policy for smaller machines.
MAX_RUNTIME_PEAK_RSS_MB = 18 * 1024
MAX_SUPPORTED_REPLY_WORDS = 256
MIN_STRESS_CASES = 1
RUNTIME_REPORT_SCHEMA_VERSION = 2
_METRIC_KEYS = (
    "first_generation_seconds",
    "load_seconds",
    "peak_rss_mb",
    "warm_mean_seconds",
    "warm_p95_seconds",
)


def runtime_machine_sha256() -> str:
    """Return a private stable binding for the machine that ran the benchmark."""

    identity = f"{platform.node()}\0{platform.machine()}".encode()
    return hashlib.sha256(identity).hexdigest()


def runtime_software_versions() -> dict[str, str]:
    """Return the decoder/runtime versions that benchmark evidence depends on."""

    try:
        mlx_lm_version = package_version("mlx-lm")
    except PackageNotFoundError:
        mlx_lm_version = "unavailable"
    return {
        "mlx_lm": mlx_lm_version,
        "python": platform.python_version(),
    }


def write_runtime_report(
    report_path: str | Path,
    *,
    model_version: str,
    adapter_sha256: str,
    candidate_sha256: str,
    iterations: int,
    statuses: Mapping[str, int],
    metrics: Mapping[str, float],
    workload: Mapping[str, int],
) -> dict[str, object]:
    """Atomically record aggregate benchmark evidence for one candidate and machine."""

    path = Path(report_path).expanduser()
    if inside_git_worktree(path):
        raise ValueError("runtime evidence must live outside a Git worktree")
    _ensure_private_parent(path)
    if (
        not model_version
        or not re.fullmatch(r"[0-9a-f]{64}", adapter_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256)
        or type(iterations) is not int
        or iterations <= 0
    ):
        raise ValueError("runtime evidence candidate binding is invalid")
    if set(statuses) - set(RUNTIME_STATUS_KEYS) or any(
        type(value) is not int or value < 0 for value in statuses.values()
    ):
        raise ValueError("runtime evidence statuses are invalid")
    normalized_statuses = {key: int(statuses.get(key, 0)) for key in RUNTIME_STATUS_KEYS}
    if sum(normalized_statuses.values()) != iterations:
        raise ValueError("runtime evidence statuses do not match iterations")
    if set(metrics) != set(_METRIC_KEYS) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in metrics.values()
    ):
        raise ValueError("runtime evidence metrics are invalid")
    if (
        set(workload) != {"max_reply_words", "stress_cases"}
        or any(type(value) is not int or value < 0 for value in workload.values())
        or workload["max_reply_words"] != MAX_SUPPORTED_REPLY_WORDS
        or workload["stress_cases"] < MIN_STRESS_CASES
    ):
        raise ValueError("runtime evidence workload is invalid")
    rendered_rate = normalized_statuses["rendered"] / iterations
    ok = bool(
        iterations >= MIN_RUNTIME_ITERATIONS
        and rendered_rate >= MIN_RUNTIME_RENDER_RATE
        and float(metrics["peak_rss_mb"]) <= MAX_RUNTIME_PEAK_RSS_MB
    )
    payload: dict[str, object] = {
        "adapter_sha256": adapter_sha256,
        "candidate_sha256": candidate_sha256,
        "iterations": iterations,
        "machine_sha256": runtime_machine_sha256(),
        "metrics": {key: float(metrics[key]) for key in _METRIC_KEYS},
        "model_version": model_version,
        "ok": ok,
        "renderer_contract_version": RENDERER_CONTRACT_VERSION,
        "rendered_rate": round(rendered_rate, 6),
        "schema_version": RUNTIME_REPORT_SCHEMA_VERSION,
        "software_versions": runtime_software_versions(),
        "statuses": dict(sorted(normalized_statuses.items())),
        "workload": dict(sorted(workload.items())),
    }
    write_private_json(path, payload)
    return payload


def runtime_report_eligible(
    report_path: str | Path,
    *,
    model_version: str,
    adapter_sha256: str,
    candidate_sha256: str,
) -> bool:
    """Validate benchmark evidence against the candidate and current machine."""

    path = Path(report_path).expanduser()
    if (
        inside_git_worktree(path)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_mode & 0o077
        or path.parent.stat().st_mode & 0o077
    ):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    required = {
        "adapter_sha256",
        "candidate_sha256",
        "iterations",
        "machine_sha256",
        "metrics",
        "model_version",
        "ok",
        "renderer_contract_version",
        "rendered_rate",
        "schema_version",
        "software_versions",
        "statuses",
        "workload",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or not isinstance(payload.get("statuses"), dict)
        or set(payload["statuses"]) != set(RUNTIME_STATUS_KEYS)
        or not isinstance(payload.get("metrics"), dict)
        or set(payload["metrics"]) != set(_METRIC_KEYS)
        or not isinstance(payload.get("workload"), dict)
        or set(payload["workload"]) != {"max_reply_words", "stress_cases"}
        or not isinstance(payload.get("software_versions"), dict)
        or set(payload["software_versions"]) != {"mlx_lm", "python"}
        or any(
            not isinstance(value, str) or not value
            for value in payload["software_versions"].values()
        )
        or type(payload.get("schema_version")) is not int
        or type(payload.get("renderer_contract_version")) is not int
        or type(payload.get("iterations")) is not int
        or isinstance(payload.get("rendered_rate"), bool)
        or not isinstance(payload.get("rendered_rate"), (int, float))
        or not math.isfinite(float(payload["rendered_rate"]))
        or any(type(value) is not int for value in payload["statuses"].values())
        or any(value < 0 for value in payload["statuses"].values())
        or any(type(value) is not int for value in payload["workload"].values())
        or any(value < 0 for value in payload["workload"].values())
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in payload["metrics"].values()
        )
    ):
        return False
    iterations = payload["iterations"]
    statuses = payload["statuses"]
    workload = payload["workload"]
    if iterations <= 0 or sum(statuses.values()) != iterations:
        return False
    rendered_rate = statuses["rendered"] / iterations
    recomputed_ok = bool(
        iterations >= MIN_RUNTIME_ITERATIONS
        and rendered_rate >= MIN_RUNTIME_RENDER_RATE
        and float(payload["metrics"]["peak_rss_mb"]) <= MAX_RUNTIME_PEAK_RSS_MB
    )
    return bool(
        payload.get("schema_version") == RUNTIME_REPORT_SCHEMA_VERSION
        and payload.get("renderer_contract_version") == RENDERER_CONTRACT_VERSION
        and payload.get("model_version") == model_version
        and payload.get("adapter_sha256") == adapter_sha256
        and payload.get("candidate_sha256") == candidate_sha256
        and payload.get("machine_sha256") == runtime_machine_sha256()
        and payload.get("software_versions") == runtime_software_versions()
        and payload.get("rendered_rate") == round(rendered_rate, 6)
        and workload["max_reply_words"] == MAX_SUPPORTED_REPLY_WORDS
        and workload["stress_cases"] >= MIN_STRESS_CASES
        and payload.get("ok") is True
        and recomputed_ok
    )


def _ensure_private_parent(path: Path) -> None:
    existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if existed and path.parent.stat().st_mode & 0o077:
        raise PermissionError("runtime evidence directory must be private (0700)")
    path.parent.chmod(0o700)
    if path.exists() and (path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077):
        raise PermissionError("runtime evidence must be a private regular file")
