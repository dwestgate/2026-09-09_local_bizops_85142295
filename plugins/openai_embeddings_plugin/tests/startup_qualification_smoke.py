#!/usr/bin/env python3
"""Offline regression smoke for embedding startup qualification."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ananta" / "src"))
sys.path.insert(0, str(REPO_ROOT / "plugins" / "openai_embeddings_plugin" / "src"))

from openai_embeddings_plugin.constants import ErrorCode  # noqa: E402
from openai_embeddings_plugin.plugin import OpenAIEmbeddingsPlugin  # noqa: E402
from openai_embeddings_plugin.response_builders import error_result  # noqa: E402

_passed = 0
_failed: list[str] = []


def _check(condition: object, label: str) -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed.append(label)
        print(f"  FAIL  {label}")


class _AddressBookService:
    @staticmethod
    def resolve_with_secrets(_name: str) -> dict[str, Any]:
        return {
            "action_status": "completed",
            "data": {
                "entries": [
                    {"field_type": "base_url", "value": "http://embeddings.test/v1"},
                    {"field_type": "model", "value": "embed-test"},
                    {"field_type": "timeout_seconds", "value": "12"},
                ]
            },
        }


class _Orchestrator:
    @staticmethod
    def get_service(name: str) -> _AddressBookService | None:
        return _AddressBookService() if name == "address_book_service" else None


def test_qualified_model_marks_plugin_ready() -> None:
    plugin = OpenAIEmbeddingsPlugin()
    plugin.orchestrator_ref = _Orchestrator()  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    def _success(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], None]:
        calls.append((url, payload))
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "model": "embed-test"}, None

    plugin._call_embeddings_api = _success  # type: ignore[method-assign]
    plugin.prepare_for_readiness()

    _check(plugin.is_ready(), "successful configured-model probe marks the plugin ready")
    _check(plugin._initialized, "plugin is initialized only after qualification succeeds")
    _check(
        calls == [
            (
                "http://embeddings.test/v1/embeddings",
                {"model": "embed-test", "input": ["startup readiness probe"]},
            )
        ],
        "qualification sends one embedding request to the configured model",
    )


def test_unreachable_model_fails_before_ready() -> None:
    plugin = OpenAIEmbeddingsPlugin()
    plugin.orchestrator_ref = _Orchestrator()  # type: ignore[assignment]
    plugin._call_embeddings_api = lambda _url, _payload: (  # type: ignore[method-assign]
        None,
        error_result(ErrorCode.CONNECTION_FAILED, "connection refused"),
    )

    raised: RuntimeError | None = None
    try:
        plugin.prepare_for_readiness()
    except RuntimeError as exc:
        raised = exc

    _check(
        raised is not None,
        "unreachable configured embedding model aborts readiness preparation",
    )
    _check(
        raised is not None and "connection refused" in str(raised),
        "qualification failure preserves the endpoint error",
    )
    _check(not plugin.is_ready(), "failed qualification cannot mark the plugin ready")
    _check(not plugin._initialized, "failed qualification leaves the plugin uninitialized")


def main() -> int:
    print("=== startup_qualification_smoke ===")
    test_qualified_model_marks_plugin_ready()
    test_unreachable_model_fails_before_ready()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
