# ---------------------------------------------------------------------------
# 3. Machinery Contract
# ---------------------------------------------------------------------------


from typing import Dict, Optional, Sequence
import numpy as np
import os
import csv
import json

from contracts.helpers import _is_finite_array, ViolationLogger

class MachineryContract:
    NAME = "Machinery"
    MESSAGES = {
        "A1": "Missing or invalid shaft speed from throttle controller.",
        "A2": "Machinery mode switching violates minimum dwell time.",
        "G1": "Thrust force from machinery is invalid.",
        "G2": "Power demand exceeds available machinery power.",
        "G3": "Thrust or power change rate exceeds allowed limits.",
    }

    def __init__(
        self,
        shaft_speed: np.ndarray,
        mode_sequence: Sequence[int],
        mode_switch_times: Sequence[float],
        dwell_time_min: float,
        thrust_force: np.ndarray,
        power_demand: np.ndarray,
        power_available: np.ndarray,
        rate_limit_thrust: float,
        rate_limit_power: float,
    ):
        self.shaft_speed = shaft_speed
        self.mode_sequence = mode_sequence
        self.mode_switch_times = mode_switch_times
        self.dwell_time_min = dwell_time_min
        self.thrust_force = thrust_force
        self.power_demand = power_demand
        self.power_available = power_available
        self.rate_limit_thrust = rate_limit_thrust
        self.rate_limit_power = rate_limit_power
        self.contract_status = {"A1": None, "A2": None, "G1": None, "G2": None, "G3": None}

    def check_A1(self):
        self.contract_status["A1"] = _is_finite_array(self.shaft_speed)
        return self.contract_status["A1"]

    def check_A2(self):
        times = np.asarray(self.mode_switch_times)
        if times.size <= 1:
            ok_dwell = True
        else:
            dt = np.diff(times)
            ok_dwell = bool(np.all(dt >= self.dwell_time_min))
        self.contract_status["A2"] = ok_dwell
        return self.contract_status["A2"]

    def check_G1(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G1"] = None
        #     return None
        self.contract_status["G1"] = _is_finite_array(self.thrust_force)
        return self.contract_status["G1"]

    def check_G2(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G2"] = None
        #     return None
        self.contract_status["G2"] = bool(np.all(self.power_demand <= self.power_available))
        return self.contract_status["G2"]

    def check_G3(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G3"] = None
        #     return None

        def _rate_limited(x, limit):
            x = np.asarray(x)
            if x.size < 2:
                return True
            dx = np.abs(np.diff(x))
            return bool(np.all(dx <= limit))

        ok_thrust = _rate_limited(self.thrust_force, self.rate_limit_thrust)
        ok_power = _rate_limited(self.power_demand, self.rate_limit_power)
        self.contract_status["G3"] = ok_thrust and ok_power
        return self.contract_status["G3"]

    def evaluate(
        self,
        logger: Optional[ViolationLogger] = None,
        t: Optional[float] = None,
        meta: Optional[Dict] = None,
    ):
        self.check_A1()
        self.check_A2()
        self.check_G1()
        self.check_G2()
        self.check_G3()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(time=t,
                               subsystem=self.NAME,
                               contract_id=clause,
                               message=msg)

        return self.contract_status