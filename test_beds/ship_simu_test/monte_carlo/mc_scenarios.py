# mc_scenarios.py
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass(frozen=True)
class Scenario:
    seed: int
    fault_start_s: float
    fault_duration_s: float
    engine_fault_mask: Dict[str, bool]  # True=healthy, False=failed

def sample_scenario(rng: np.random.Generator, sim_time_s: float) -> Scenario:
    seed = int(rng.integers(0, 2**31 - 1))

    # sample time window
    fault_start_s = float(rng.uniform(0.1 * sim_time_s, 0.8 * sim_time_s))
    fault_duration_s = float(rng.uniform(200.0, 1200.0))  # tune

    # sample fault combo
    combos: List[Dict[str, bool]] = [
        {"ME": False, "DG1": True,  "DG2": True,  "HSG": True},   # ME fail
        {"ME": True,  "DG1": False, "DG2": True,  "HSG": True},   # DG1 fail
        {"ME": True,  "DG1": True,  "DG2": False, "HSG": True},   # DG2 fail
        {"ME": True,  "DG1": True,  "DG2": True,  "HSG": False},  # HSG fail
        {"ME": False, "DG1": False, "DG2": True,  "HSG": False},  # your current test
    ]
    probs = np.array([0.25, 0.15, 0.15, 0.10, 0.35], dtype=float)
    probs = probs / probs.sum()

    engine_fault_mask = combos[int(rng.choice(len(combos), p=probs))]

    return Scenario(
        seed=seed,
        fault_start_s=fault_start_s,
        fault_duration_s=fault_duration_s,
        engine_fault_mask=engine_fault_mask,
    )

def engine_status_at_time(s: Scenario, t: float) -> Dict[str, bool]:
    active = (t >= s.fault_start_s) and (t < s.fault_start_s + s.fault_duration_s)
    if active:
        return dict(s.engine_fault_mask)
    return {"ME": True, "DG1": True, "DG2": True, "HSG": True}
