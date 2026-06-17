# ---------------------------------------------------------------------------
# 4. Throttle Controller Contract
# ---------------------------------------------------------------------------

from typing import Optional
import numpy as np

from contracts.helpers import _is_finite_array


class ThrottleControllerContract:
    NAME = "ThrottleController"
    MESSAGES = {
        "A1": "Desired speed is invalid or outside allowable range.",
        "G1": "Propulsion command did not produce shaft response consistent with the requested speed.",
    }

    def __init__(
        self,
        desired_speed: np.ndarray,
        shaft_speed_cmd: np.ndarray,
        vmax: float,
        tolerance: float,
        actual_shaft_speed: Optional[np.ndarray] = None,
        min_shaft_speed_rpm: float = 5.0,
    ):
        self.desired_speed = desired_speed
        self.shaft_speed_cmd = shaft_speed_cmd
        self.vmax = vmax
        self.tolerance = tolerance
        self.actual_shaft_speed = actual_shaft_speed
        self.min_shaft_speed_rpm = min_shaft_speed_rpm
        self.contract_status = {"A1": None, "G1": None}

    def check_A1(self):
        v = np.asarray(self.desired_speed, dtype=float)
        finite = _is_finite_array(v)
        within_range = bool(np.all((0.0 <= v) & (v <= self.vmax)))
        self.contract_status["A1"] = finite and within_range
        return self.contract_status["A1"]

    def check_G1(self):
        v_des = np.asarray(self.desired_speed, dtype=float)
        actual = np.asarray(
            self.actual_shaft_speed if self.actual_shaft_speed is not None else self.shaft_speed_cmd,
            dtype=float,
        )

        if not _is_finite_array(actual):
            self.contract_status["G1"] = False
            return self.contract_status["G1"]

        desired = float(np.nanmax(v_des)) if v_des.size else 0.0
        actual_rpm = float(np.nanmax(np.abs(actual))) if actual.size else 0.0

        if desired <= self.tolerance:
            self.contract_status["G1"] = bool(actual_rpm <= self.min_shaft_speed_rpm)
        else:
            self.contract_status["G1"] = bool(actual_rpm >= self.min_shaft_speed_rpm)
        return self.contract_status["G1"]

    def evaluate(self, logger=None, t=None, meta=None):
        self.check_A1()
        self.check_G1()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    message = self.MESSAGES.get(
                        clause,
                        f"{self.NAME} contract {clause} violated.",
                    )
                    logger.log(
                        time=t,
                        subsystem=self.NAME,
                        contract_id=clause,
                        message=message,
                    )

        return self.contract_status
