from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from summarize_bagel_loop_t2i_multiseed import summarize_multiseed  # noqa: E402


def _summary(*runs):
    return {
        "runs": [
            {
                "name": name,
                "overall": {
                    "soft_tifa_am": am,
                    "soft_tifa_gm": gm,
                    "atom_weighted_am": atom_am,
                },
            }
            for name, am, gm, atom_am in runs
        ]
    }


def test_multiseed_summary_separates_z0_variability_from_loop_delta():
    main = _summary(("Z0", 60.0, 10.0, 65.0), ("Z3", 62.0, 13.0, 67.0))
    seed43 = _summary(("Z0", 61.0, 11.0, 66.0))
    seed44 = _summary(("Z0", 59.0, 9.0, 64.0))

    result = summarize_multiseed(main, [(43, seed43), (44, seed44)])

    assert [row["seed"] for row in result["z0_seeds"]] == [42, 43, 44]
    assert result["z0_variability"]["soft_tifa_am"]["mean"] == 60.0
    assert result["z0_variability"]["soft_tifa_am"]["sample_std"] == 1.0
    z3 = result["loop_arms"][0]
    assert z3["soft_tifa_am"]["delta_vs_z0_same_seed"] == 2.0
    assert z3["soft_tifa_am"]["outside_z0_seed_range"] is True
    assert z3["soft_tifa_gm"]["delta_vs_z0_seed_mean"] == pytest.approx(3.0)
