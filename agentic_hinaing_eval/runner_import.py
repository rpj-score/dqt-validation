from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .adapter import HinaingImportAdapter
from .models import ValidationScenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Subprocess runner for one Hinaing fixture scenario.")
    parser.add_argument("--hinaing-root", type=Path, required=True)
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument("--target-stage", type=str, default=None)
    parser.add_argument("--patch", type=str, default="drop")
    args = parser.parse_args()

    scenario = ValidationScenario.from_dict(json.loads(input()))
    adapter = HinaingImportAdapter(args.hinaing_root)

    if args.counterfactual and args.target_stage:
        from .agent_influence import PATCH_STRATEGIES

        patch_fn = PATCH_STRATEGIES.get(args.patch)
        if patch_fn is None:
            raise ValueError(f"Unknown patch strategy: {args.patch!r}; valid: {sorted(PATCH_STRATEGIES)}")
        artifact = asyncio.run(
            adapter.run_fixture_with_patch(scenario, args.target_stage, patch_fn)
        )
    else:
        artifact = asyncio.run(adapter.run_fixture(scenario))

    print(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
