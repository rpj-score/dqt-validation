from __future__ import annotations

from typing import Any


def _steps_to_openai_messages(steps: list[str]) -> list[dict[str, Any]]:
    """Represent graph/node steps as OpenAI-style tool-call messages."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": step, "arguments": "{}"},
                }
            ],
        }
        for idx, step in enumerate(steps, start=1)
    ]


def agentevals_trajectory_score(actual: list[str], expected: list[str]) -> dict[str, Any]:
    """Use LangChain AgentEvals when installed, otherwise use deterministic fallback."""
    if not expected:
        return {"score": 1.0, "comment": "No reference trajectory supplied.", "missing": []}
    try:
        from agentevals.trajectory.match import create_trajectory_match_evaluator  # type: ignore

        evaluator = create_trajectory_match_evaluator(
            trajectory_match_mode="superset",
            tool_args_match_mode="ignore",
        )
        result = evaluator(
            outputs=_steps_to_openai_messages(actual),
            reference_outputs=_steps_to_openai_messages(expected),
        )
        raw_score = result.get("score", 0.0) if isinstance(result, dict) else 0.0
        numeric = 1.0 if raw_score is True else 0.0 if raw_score is False else float(raw_score)
        fallback = deterministic_trajectory_score(actual, expected)
        return {
            "score": numeric,
            "comment": result.get("comment") if isinstance(result, dict) else "AgentEvals trajectory evaluator completed.",
            "missing": fallback.get("missing", []),
            "evaluator": "agentevals",
        }
    except Exception as exc:
        fallback = deterministic_trajectory_score(actual, expected)
        fallback["evaluator"] = "deterministic_fallback"
        fallback["comment"] = f"{fallback['comment']} AgentEvals unavailable or failed: {exc}"
        return fallback


def deterministic_trajectory_score(actual: list[str], expected: list[str]) -> dict[str, Any]:
    """Fallback trajectory evaluator compatible with the AgentEvals score shape.

    This is intentionally simple and transparent. If the optional `agentevals`
    package is available, the harness can layer its evaluators on top, but this
    fallback keeps the validation framework runnable in offline thesis settings.
    """
    if not expected:
        return {"score": 1.0, "comment": "No reference trajectory supplied.", "missing": []}
    cursor = 0
    matched = 0
    missing: list[str] = []
    for step in expected:
        try:
            found_at = actual.index(step, cursor)
        except ValueError:
            missing.append(step)
            continue
        matched += 1
        cursor = found_at + 1
    score = matched / max(len(expected), 1)
    return {
        "score": score,
        "comment": f"Matched {matched}/{len(expected)} expected trajectory steps.",
        "missing": missing,
        "evaluator": "deterministic_fallback",
    }


def optional_agentevals_status() -> dict[str, Any]:
    try:
        import agentevals  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "detail": f"agentevals unavailable; deterministic trajectory scoring will be used ({exc}).",
        }
    return {
        "available": True,
        "detail": f"agentevals import available from {getattr(agentevals, '__file__', 'unknown')}",
    }
