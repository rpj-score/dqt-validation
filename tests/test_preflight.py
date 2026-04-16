from pathlib import Path

from agentic_hinaing_eval.preflight import run_preflight


def test_preflight_reports_required_paths() -> None:
    result = run_preflight(Path("../Hinaing"))
    check_ids = {check["id"] for check in result["checks"]}
    assert "graph_entrypoint" in check_ids
    assert "frontend_snapshot_route" in check_ids
    assert "query_orchestrator_signature" in check_ids


def test_preflight_detects_current_query_orchestrator_signature_gap() -> None:
    result = run_preflight(Path("../Hinaing"))
    signature = [check for check in result["checks"] if check["id"] == "query_orchestrator_signature"][0]
    assert signature["passed"] is False

