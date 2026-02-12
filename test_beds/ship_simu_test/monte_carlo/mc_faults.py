# mc_faults.py
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

HEALTHY = {"ME": True, "DG1": True, "DG2": True, "HSG": True}

@dataclass(frozen=True)
class FaultScenario:
    seed: int
    start_s: float
    duration_s: float
    fault_mask: Dict[str, bool]  # True=healthy, False=failed

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def active(self, t: float) -> bool:
        return (t >= self.start_s) and (t < self.end_s)

    def engine_status(self, t: float) -> Dict[str, bool]:
        return dict(self.fault_mask) if self.active(t) else dict(HEALTHY)

def sample_fault_scenario(
    rng: np.random.Generator,
    sim_time_s: float,
    start_frac=(0.1, 0.8),
    duration_range_s=(200.0, 1200.0),
) -> FaultScenario:
    seed = int(rng.integers(0, 2**31 - 1))

    start_s = float(rng.uniform(start_frac[0] * sim_time_s, start_frac[1] * sim_time_s))
    duration_s = float(rng.uniform(duration_range_s[0], duration_range_s[1]))

    # Define your fault combinations here (add more as you like)
    combos: List[Dict[str, bool]] = [
        {"ME": False, "DG1": True,  "DG2": True,  "HSG": True},   # ME fail 0111
        {"ME": False, "DG1": False,  "DG2": True,  "HSG": True},   # ME fail 2 0011
        {"ME": False, "DG1": False,  "DG2": False,  "HSG": True},   # ME fail 3 0001
        {"ME": True,  "DG1": False, "DG2": True,  "HSG": True},   # DG1 fail 1011
        {"ME": True,  "DG1": False, "DG2": False,  "HSG": True},   # DG1 fail 2 1001
        {"ME": True,  "DG1": False, "DG2": False,  "HSG": False},   # DG1 fail 3 1000
        {"ME": True,  "DG1": True,  "DG2": False, "HSG": True},   # DG2 fail 1101
        {"ME": True,  "DG1": True,  "DG2": False, "HSG": False},   # DG2 fail 2 1100
        {"ME": True,  "DG1": True,  "DG2": True,  "HSG": False},  # HSG fail 1110
        {"ME": False, "DG1": False, "DG2": False,  "HSG": False},  # example combo 0000
        {"ME": False, "DG1": True, "DG2": True,  "HSG": False},  # example combo 0110
        {"ME": True, "DG1": False, "DG2": True,  "HSG": False},  # example combo 1010
        {"ME": False, "DG1": False, "DG2": True,  "HSG": False},  # example combo 0010
        {"ME": False, "DG1": True, "DG2": False,  "HSG": False},  # example combo 0100
        {"ME": False, "DG1": True, "DG2": False,  "HSG": True},  # example combo 0101
    ]
    # probs = np.array([0.25, 0.15, 0.15, 0.10, 0.35], dtype=float) #or just use this to update the probability of occurence
    probs = np.ones(len(combos), dtype=float)
    probs /= probs.sum()

    fault_mask = combos[int(rng.choice(len(combos), p=probs))]

    return FaultScenario(seed=seed, start_s=start_s, duration_s=duration_s, fault_mask=fault_mask)
