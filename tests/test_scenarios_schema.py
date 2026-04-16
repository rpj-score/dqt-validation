from pathlib import Path

from agentic_hinaing_eval.adapter import HINAING_STAGES
from agentic_hinaing_eval.io import load_scenarios
from agentic_hinaing_eval.models import SCENARIO_FAMILIES


SCENARIO_DIR = Path("data/scenarios")

# The old, hardcoded reference trajectory that the previous eval framework
# used as both actual and expected. Any scenario still carrying this list is a
# regression — trajectories must be the Hinaing-emitted stage names.
LEGACY_TRAJECTORY = [
    "orchestrate_queries",
    "fetch_documents",
    "retrieve_internal_knowledge",
    "label_sentiment_and_analyze",
    "consolidate_memory",
    "theme_agents",
    "build_snapshot",
]


def test_scenarios_load_without_error() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    assert scenarios, "No scenarios loaded — data/scenarios should contain JSONL files."
    assert len(scenarios) >= 40, f"Expected ~40 scenarios, loaded {len(scenarios)}."


def test_every_scenario_has_a_valid_family() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    for scenario in scenarios:
        assert scenario.family in SCENARIO_FAMILIES, (
            f"Scenario {scenario.id} has family={scenario.family!r}; "
            f"expected one of {SCENARIO_FAMILIES}"
        )


def test_no_scenario_uses_the_legacy_fabricated_trajectory() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    for scenario in scenarios:
        assert scenario.reference_trajectory != LEGACY_TRAJECTORY, (
            f"Scenario {scenario.id} still uses the legacy fabricated trajectory "
            f"instead of the real Hinaing stage names."
        )


def test_reference_trajectories_are_subsets_of_hinaing_stages() -> None:
    valid = set(HINAING_STAGES)
    scenarios = load_scenarios(SCENARIO_DIR)
    for scenario in scenarios:
        for step in scenario.reference_trajectory:
            assert step in valid, (
                f"Scenario {scenario.id} references unknown stage {step!r}; "
                f"Hinaing emits only {sorted(valid)}"
            )


def test_cache_family_has_warmup_refs() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    scenario_ids = {s.id for s in scenarios}
    cache_followups = [s for s in scenarios if s.family == "cache" and "warm_cache" in s.tags]
    assert cache_followups, "cache family should contain warm_cache follow-ups"
    for scenario in cache_followups:
        assert scenario.warmup_ref, f"{scenario.id} missing warmup_ref"
        assert scenario.warmup_ref in scenario_ids, (
            f"{scenario.id}.warmup_ref={scenario.warmup_ref} not found in suite"
        )


def test_adversarial_family_has_semantic_forbidden() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    adversarial = [s for s in scenarios if s.family == "adversarial"]
    assert adversarial, "adversarial family should be populated"
    for scenario in adversarial:
        assert scenario.semantic_forbidden or scenario.expected.get("must_not_claim"), (
            f"{scenario.id} needs at least one paraphrase-resistant forbidden item"
        )


def test_ablation_pairs_cover_full_and_ablated_presets() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    ablation = [s for s in scenarios if s.family == "ablation"]
    by_id = {s.id: s for s in ablation}
    seen_pairs = set()
    for scenario in ablation:
        pair = scenario.ablation_pair
        assert pair, f"{scenario.id} should declare ablation_pair"
        full_id = pair["full_id"]
        ablated_id = pair["ablated_id"]
        seen_pairs.add((full_id, ablated_id))
        assert full_id in by_id and ablated_id in by_id
        assert by_id[full_id].request.get("ablation_preset") == "full"
        assert by_id[ablated_id].request.get("ablation_preset") == "ablated"
    assert seen_pairs, "ablation family should define at least one pair"
