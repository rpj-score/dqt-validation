import asyncio

from agentic_hinaing_eval.adapter import _ProgressRecorder, HINAING_STAGES


def test_progress_recorder_captures_emitted_stages_in_order() -> None:
    recorder = _ProgressRecorder()
    stages = [("query_orchestrator", "start", 0.1), ("retrieval", "done", 0.3), ("snapshot", "done", 1.0)]

    async def drive() -> None:
        for stage, message, progress in stages:
            await recorder(stage, message, progress)
        recorder.finalize()

    asyncio.run(drive())
    assert recorder.trajectory == ["query_orchestrator", "retrieval", "snapshot"]
    assert recorder.node_order == ["query_orchestrator", "retrieval", "snapshot"]
    assert len(recorder.tool_events) == 3
    for event in recorder.tool_events:
        assert "latency_ms" in event


def test_progress_recorder_dedupes_repeated_stages_in_node_order() -> None:
    recorder = _ProgressRecorder()

    async def drive() -> None:
        await recorder("analyze", "m1", 0.4)
        await recorder("analyze", "m2", 0.45)
        await recorder("snapshot", "done", 1.0)
        recorder.finalize()

    asyncio.run(drive())
    assert recorder.trajectory == ["analyze", "analyze", "snapshot"]
    assert recorder.node_order == ["analyze", "snapshot"]


def test_hinaing_stages_constant_matches_graph_names() -> None:
    assert HINAING_STAGES == [
        "query_orchestrator",
        "retrieval",
        "recall",
        "analyze",
        "memory",
        "themes",
        "snapshot",
    ]
