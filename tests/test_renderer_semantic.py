from personal_voice.renderer import format_renderer_target
from personal_voice.renderer_semantic import evaluate_semantic_gates


class IdentityBackend:
    version = "identity"

    def generate(self, _medium: str, draft: str) -> str:
        return format_renderer_target(draft)


class CommitmentAddingBackend:
    version = "unsafe"

    def generate(self, _medium: str, draft: str) -> str:
        return format_renderer_target(f"{draft} I will definitely take care of it.")


def test_semantic_gates_pass_identity_and_reject_added_commitments() -> None:
    safe = evaluate_semantic_gates(IdentityBackend())
    unsafe = evaluate_semantic_gates(CommitmentAddingBackend())

    assert safe["ok"] is True
    assert safe["passed"] == safe["total"] == 5
    assert unsafe["ok"] is False
    assert unsafe["failed"] >= 1
    assert all("text" not in result for result in unsafe["results"])
