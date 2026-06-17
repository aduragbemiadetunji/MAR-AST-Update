from dataclasses import dataclass
from typing import Dict, List, Optional
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


@dataclass(frozen=True)
class FaultEvent:
    start_s: float
    duration_s: float
    fault_mask: Dict[str, bool]  # True=healthy, False=failed

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def active(self, t: float) -> bool:
        return (t >= self.start_s) and (t < self.end_s)


class MultiFaultScenario:
    def __init__(self, seed: int, events: List[FaultEvent]):
        self.seed = seed
        self.events = sorted(events, key=lambda e: e.start_s)

    def engine_status(self, t: float) -> Dict[str, bool]:
        """
        Combine all active faults at time t.
        If any active event disables an engine, that engine becomes unavailable.
        """
        status = dict(HEALTHY)

        for ev in self.events:
            if ev.active(t):
                for k in status.keys():
                    status[k] = status[k] and ev.fault_mask[k]

        return status

    def active_events(self, t: float) -> List[FaultEvent]:
        return [ev for ev in self.events if ev.active(t)]

    def is_fault_active(self, t: float) -> bool:
        return len(self.active_events(t)) > 0

    @property
    def start_s(self) -> float:
        if not self.events:
            return np.nan
        return self.events[0].start_s

    @property
    def duration_s(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].end_s - self.events[0].start_s

    @property
    def end_s(self) -> float:
        if not self.events:
            return np.nan
        return self.events[-1].end_s


def _fault_combos() -> List[Dict[str, bool]]:
    """
    True = healthy, False = failed
    """
    return [
        {"ME": False, "DG1": True,  "DG2": True,  "HSG": True},   # ME
        {"ME": False, "DG1": False, "DG2": True,  "HSG": True},   # ME + DG1
        {"ME": False, "DG1": False, "DG2": False, "HSG": True},   # ME + DG1 + DG2
        {"ME": True,  "DG1": False, "DG2": True,  "HSG": True},   # DG1
        {"ME": True,  "DG1": False, "DG2": False, "HSG": True},   # DG1 + DG2
        {"ME": True,  "DG1": False, "DG2": False, "HSG": False},  # DG1 + DG2 + HSG
        {"ME": True,  "DG1": True,  "DG2": False, "HSG": True},   # DG2
        {"ME": True,  "DG1": True,  "DG2": False, "HSG": False},  # DG2 + HSG
        {"ME": True,  "DG1": True,  "DG2": True,  "HSG": False},  # HSG
        {"ME": False, "DG1": False, "DG2": False, "HSG": False},  # all fail
        {"ME": False, "DG1": True,  "DG2": True,  "HSG": False},  # ME + HSG
        {"ME": True,  "DG1": False, "DG2": True,  "HSG": False},  # DG1 + HSG
        {"ME": False, "DG1": False, "DG2": True,  "HSG": False},  # ME + DG1 + HSG
        {"ME": False, "DG1": True,  "DG2": False, "HSG": False},  # ME + DG2 + HSG
        {"ME": False, "DG1": True,  "DG2": False, "HSG": True},   # ME + DG2
    ]


def sample_fault_scenario(
    rng: np.random.Generator,
    sim_time_s: float,
    start_frac=(0.1, 0.8),
    duration_range_s=(200.0, 600.0),
) -> FaultScenario:
    seed = int(rng.integers(0, 2**31 - 1))

    start_s = float(rng.uniform(start_frac[0] * sim_time_s, start_frac[1] * sim_time_s))
    duration_s = float(rng.uniform(duration_range_s[0], duration_range_s[1]))

    combos = _fault_combos()
    probs = np.ones(len(combos), dtype=float)
    probs /= probs.sum()

    fault_mask = combos[int(rng.choice(len(combos), p=probs))]

    return FaultScenario(seed=seed, start_s=start_s, duration_s=duration_s, fault_mask=fault_mask)


# def sample_multi_fault_scenario(
#     rng: np.random.Generator,
#     sim_time_s: float,
#     n_faults: int = 5,
#     first_start_frac=(0.1, 0.25),
#     min_gap_s: float = 120.0,
#     max_gap_s: float = 500.0,
#     duration_range_s=(120.0, 600.0),
#     allow_overlap: bool = False,
#     combo_probs: Optional[np.ndarray] = None,
# ) -> MultiFaultScenario:
#     seed = int(rng.integers(0, 2**31 - 1))
#     combos = _fault_combos()

#     if combo_probs is None:
#         probs = np.ones(len(combos), dtype=float)
#     else:
#         probs = np.asarray(combo_probs, dtype=float)
#         if len(probs) != len(combos):
#             raise ValueError("combo_probs must have the same length as the number of fault combinations")
#     probs /= probs.sum()

#     events: List[FaultEvent] = []

#     current_t = float(rng.uniform(first_start_frac[0] * sim_time_s, first_start_frac[1] * sim_time_s))

#     for _ in range(n_faults):
#         duration_s = float(rng.uniform(duration_range_s[0], duration_range_s[1]))
#         fault_mask = combos[int(rng.choice(len(combos), p=probs))]

#         if current_t >= sim_time_s:
#             break

#         # Clip duration so event stays within the simulation
#         duration_s = min(duration_s, max(1.0, sim_time_s - current_t))

#         events.append(FaultEvent(
#             start_s=current_t,
#             duration_s=duration_s,
#             fault_mask=fault_mask
#         ))

#         if allow_overlap:
#             gap = float(rng.uniform(-0.35 * duration_s, max_gap_s))
#         else:
#             gap = float(rng.uniform(min_gap_s, max_gap_s))

#         current_t = current_t + duration_s + gap

#     return MultiFaultScenario(seed=seed, events=events)


# def sample_multi_fault_scenario(
#     rng: np.random.Generator,
#     sim_time_s: float,
#     n_faults: int = 5,
#     start_frac=(0.08, 0.92),
#     duration_range_s=(120.0, 600.0),
#     min_separation_s=800.0,
#     combo_probs: Optional[np.ndarray] = None,
# ) -> MultiFaultScenario:
#     seed = int(rng.integers(0, 2**31 - 1))
#     combos = _fault_combos()

#     if combo_probs is None:
#         probs = np.ones(len(combos), dtype=float)
#     else:
#         probs = np.asarray(combo_probs, dtype=float)
#         if len(probs) != len(combos):
#             raise ValueError("combo_probs must have the same length as the number of fault combinations")
#     probs = probs / probs.sum()

#     t_min = float(start_frac[0] * sim_time_s)
#     t_max = float(start_frac[1] * sim_time_s)

#     if t_max <= t_min:
#         raise ValueError("Invalid start_frac range")

#     # Rejection sampling for spaced starts
#     starts = []
#     max_tries = 5000
#     tries = 0

#     while len(starts) < n_faults and tries < max_tries:
#         s = float(rng.uniform(t_min, t_max))
#         if all(abs(s - s_old) >= min_separation_s for s_old in starts):
#             starts.append(s)
#         tries += 1

#     starts.sort()

#     events: List[FaultEvent] = []
#     for s in starts:
#         duration_s = float(rng.uniform(duration_range_s[0], duration_range_s[1]))
#         duration_s = min(duration_s, max(1.0, sim_time_s - s))

#         fault_mask = combos[int(rng.choice(len(combos), p=probs))]

#         events.append(FaultEvent(
#             start_s=s,
#             duration_s=duration_s,
#             fault_mask=fault_mask
#         ))

#     return MultiFaultScenario(seed=seed, events=events)

def sample_multi_fault_scenario(
    rng: np.random.Generator,
    sim_time_s: float,
    n_faults: int = 5,
    start_frac=(0.08, 0.92),
    duration_range_s=(120.0, 600.0),
    combo_probs: Optional[np.ndarray] = None,
) -> MultiFaultScenario:
    seed = int(rng.integers(0, 2**31 - 1))
    combos = _fault_combos()

    if combo_probs is None:
        probs = np.ones(len(combos), dtype=float)
    else:
        probs = np.asarray(combo_probs, dtype=float)
        if len(probs) != len(combos):
            raise ValueError("combo_probs must have the same length as the number of fault combinations")
    probs = probs / probs.sum()

    t_min = float(start_frac[0] * sim_time_s)
    t_max = float(start_frac[1] * sim_time_s)

    edges = np.linspace(t_min, t_max, n_faults + 1)

    events: List[FaultEvent] = []
    for i in range(n_faults):
        left = edges[i]
        right = edges[i + 1]

        s = float(rng.uniform(left, right))
        duration_s = float(rng.uniform(duration_range_s[0], duration_range_s[1]))
        duration_s = min(duration_s, max(1.0, sim_time_s - s))

        fault_mask = combos[int(rng.choice(len(combos), p=probs))]

        events.append(FaultEvent(
            start_s=s,
            duration_s=duration_s,
            fault_mask=fault_mask
        ))

    return MultiFaultScenario(seed=seed, events=events)