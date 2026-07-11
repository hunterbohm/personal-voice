"""Deterministic adversarial semantic gates for renderer promotion."""

from __future__ import annotations

import re

from personal_voice.renderer import RendererBackend, render_draft

_CASES = (
    {
        "draft": "I cannot meet before 4:30 on Friday.",
        "forbidden": (r"\bi will\b", r"\bi'll\b", r"\bconfirmed\b"),
        "medium": "text",
        "name": "negated_availability",
        "preserve": ("cannot", "before", "Friday"),
    },
    {
        "draft": "I might send the revised proposal by Monday, but I am not promising that yet.",
        "forbidden": (
            r"\bi will\b",
            r"\bi'll\b",
            r"\bdefinitely\b",
            r"\bguarantee(?:d|s)?\b",
            r"\bconfirmed\b",
        ),
        "medium": "email",
        "name": "uncertain_commitment",
        "preserve": ("might", "Monday", "not promising"),
    },
    {
        "draft": 'Can you review "Project Atlas" by 2 PM?',
        "forbidden": (r"\bi will\b", r"\bi'll\b", r"\bdone\b", r"\bapproved\b"),
        "medium": "dm",
        "name": "question_and_quoted_name",
        "preserve": ("Can you", "2 PM"),
    },
    {
        "draft": "Do not publish https://example.com/review or share it with @alex.",
        "forbidden": (r"\bgo ahead\b", r"\bfeel free\b", r"\bi will\b", r"\bi'll\b"),
        "medium": "public",
        "name": "prohibited_action",
        "preserve": ("Do not", "or share"),
    },
    {
        "draft": "$1,250 is the maximum budget, not the approved price.",
        "forbidden": (
            r"\bgo ahead\b",
            r"\bapproved budget\b",
            r"\bthe approved price is\b",
        ),
        "medium": "email",
        "name": "money_not_approval",
        "preserve": ("maximum", "not the approved price"),
    },
)


def evaluate_semantic_gates(backend: RendererBackend) -> dict[str, object]:
    """Exercise high-risk semantic shapes and return aggregate-only results."""

    results: list[dict[str, object]] = []
    for case in _CASES:
        result = render_draft(
            str(case["draft"]),
            str(case["medium"]),
            backend,
            preserve_literals=case["preserve"],
        )
        rendered = str(result["text"])
        reason = str(result["reason_code"])
        passed = result["status"] == "rendered"
        if passed and rendered.count("?") != str(case["draft"]).count("?"):
            passed = False
            reason = "question_shape"
        if passed and any(
            re.search(pattern, rendered, re.IGNORECASE) for pattern in case["forbidden"]
        ):
            passed = False
            reason = "forbidden_commitment"
        results.append(
            {
                "name": str(case["name"]),
                "passed": passed,
                "reason_code": reason,
                "status": str(result["status"]),
            }
        )
    failed = sum(not bool(result["passed"]) for result in results)
    return {
        "failed": failed,
        "ok": failed == 0,
        "passed": len(results) - failed,
        "results": results,
        "total": len(results),
    }
