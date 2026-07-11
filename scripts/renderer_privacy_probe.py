#!/usr/bin/env python3
"""Run schema-two aggregate privacy evidence for tuned and untouched base."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.privacy_evidence import (  # noqa: E402
    PrivacyTrainingPair,
    build_privacy_evidence_index,
    observe_privacy_evidence,
    unavailable_privacy_observation,
    write_privacy_report,
)
from personal_voice.privacy_probe import (  # noqa: E402
    MAX_PRIVACY_PROBES,
    privacy_attack,
    privacy_report_eligible,
    require_candidate_dataset_binding,
)
from personal_voice.renderer import (  # noqa: E402
    RENDER_SYSTEM_PROMPT,
    load_mlx_inprocess_backend,
    parse_renderer_target,
    parse_renderer_user_prompt,
)

_MEDIA = ("text", "dm", "email", "public")
_PUBLIC_REPORT_KEYS = (
    "dataset_records",
    "evidence_kind",
    "model_version",
    "ok",
    "paired_tuned_base",
    "probes",
    "renderer_contract_version",
    "schema_version",
    "software_versions",
    "variants",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=MAX_PRIVACY_PROBES)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.count != MAX_PRIVACY_PROBES:
        raise SystemExit(
            f"schema-two privacy evidence requires exactly {MAX_PRIVACY_PROBES} probes"
        )
    data = args.data.expanduser()
    try:
        dataset_sha256, dataset_records = require_candidate_dataset_binding(args.manifest, data)
    except ValueError as error:
        raise SystemExit("private candidate dataset binding is invalid") from error
    evidence_index = build_privacy_evidence_index(_load_pairs(data))
    control = tuple(
        observe_privacy_evidence(
            index=evidence_index,
            probe_index=index,
            raw_output=privacy_attack(index),
            attack_prompt=privacy_attack(index),
            system_prompt=RENDER_SYSTEM_PROMPT,
        )
        for index in range(MAX_PRIVACY_PROBES)
    )

    tuned_backend = load_mlx_inprocess_backend(args.manifest)
    binding = (
        tuned_backend.version,
        tuned_backend.artifact_hash,
        tuned_backend.candidate_hash,
    )
    tuned = _run_probes(tuned_backend, evidence_index)
    del tuned_backend
    gc.collect()

    base_backend = load_mlx_inprocess_backend(args.manifest, use_adapter=False)
    if (
        base_backend.version,
        base_backend.artifact_hash,
        base_backend.candidate_hash,
    ) != binding:
        raise SystemExit("privacy evidence backends are not candidate-bound")
    base = _run_probes(base_backend, evidence_index)
    del base_backend
    gc.collect()

    try:
        final_sha256, final_records = require_candidate_dataset_binding(args.manifest, data)
    except ValueError as error:
        raise SystemExit("private candidate dataset binding changed during probing") from error
    if final_sha256 != dataset_sha256 or final_records != dataset_records:
        raise SystemExit("private candidate dataset binding changed during probing")
    model_version, adapter_sha256, candidate_sha256 = binding
    report_path = (
        args.report.expanduser()
        if args.report is not None
        else args.manifest.expanduser().parent.parent
        / "evaluation/privacy"
        / f"{model_version}.json"
    )
    report = write_privacy_report(
        report_path,
        model_version=model_version,
        adapter_sha256=adapter_sha256,
        candidate_sha256=candidate_sha256,
        dataset_sha256=dataset_sha256,
        dataset_records=dataset_records,
        tuned_observations=tuned,
        base_observations=base,
        prompt_control_observations=control,
    )
    public = {key: report[key] for key in _PUBLIC_REPORT_KEYS}
    public["activation_eligible"] = privacy_report_eligible(
        report_path,
        model_version=model_version,
        adapter_sha256=adapter_sha256,
        candidate_sha256=candidate_sha256,
        dataset_sha256=dataset_sha256,
        dataset_records=dataset_records,
    )
    public["report_path"] = str(report_path)
    print(json.dumps(public, sort_keys=True))
    return 0 if report["ok"] else 1


def _load_pairs(data: Path) -> list[PrivacyTrainingPair]:
    pairs: list[PrivacyTrainingPair] = []
    for split in ("train", "valid", "test"):
        partition = "train" if split == "train" else "heldout"
        with (data / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    messages = json.loads(line)["messages"]
                    canonical = messages[-2]
                    if canonical.get("role") != "user" or not isinstance(
                        canonical.get("content"), str
                    ):
                        raise ValueError("invalid private canonical record")
                    pairs.append(
                        PrivacyTrainingPair(
                            partition,
                            parse_renderer_user_prompt(canonical["content"]),
                            parse_renderer_target(messages[-1]["content"]),
                        )
                    )
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                    raise SystemExit(
                        "private training dataset has an invalid private record"
                    ) from error
    return pairs


def _run_probes(backend, evidence_index):
    observations = []
    for index in range(MAX_PRIVACY_PROBES):
        attack = privacy_attack(index)
        try:
            raw_output = backend.generate(_MEDIA[index % len(_MEDIA)], attack)
        except Exception:
            observations.append(unavailable_privacy_observation(index))
            continue
        observations.append(
            observe_privacy_evidence(
                index=evidence_index,
                probe_index=index,
                raw_output=raw_output,
                attack_prompt=attack,
                system_prompt=RENDER_SYSTEM_PROMPT,
            )
        )
        del raw_output
    return tuple(observations)


if __name__ == "__main__":
    raise SystemExit(main())
