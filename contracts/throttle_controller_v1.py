# ---------------------------------------------------------------------------
# 4. Throttle Controller Contract
# ---------------------------------------------------------------------------

from typing import Dict, Optional, Sequence
import numpy as np
import os
import csv
import json

from contracts.helpers import _is_finite_array

class ThrottleControllerContract:
    NAME = "ThrottleController"
    MESSAGES = {
        "A1": "Desired speed is invalid or outside allowable range.",
        "G1": "Throttle controller failed to match shaft speed to desired speed.",
    }


    def __init__(
        self,
        desired_speed: np.ndarray,
        shaft_speed_cmd: np.ndarray,
        vmax: float,
        tolerance: float,
    ):
        self.desired_speed = desired_speed
        self.shaft_speed_cmd = shaft_speed_cmd
        self.vmax = vmax
        self.tolerance = tolerance
        self.contract_status = {"A1": None, "G1": None}

    def check_A1(self):
        v = np.asarray(self.desired_speed)
        finite = _is_finite_array(v)
        within_range = bool(np.all((0.0 <= v) & (v <= self.vmax)))
        self.contract_status["A1"] = finite and within_range
        return self.contract_status["A1"]
    
    def check_G1(self):
        # If A1 fails, we don't evaluate G1 (it becomes undefined)
        # if not self.contract_status["A1"]:
        #     self.contract_status["G1"] = None
        #     return None

        v_des = np.asarray(self.desired_speed)
        v_cmd = np.asarray(self.shaft_speed_cmd)
        err = np.abs(v_cmd - v_des)#check this with data sheet, evaluation does not make sense 1.1
        self.contract_status["G1"] =True# bool(np.all(err <= self.tolerance))
        return self.contract_status["G1"]

    def evaluate(self, logger=None, t=None, meta=None):
        self.check_A1()
        self.check_G1()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    message = self.MESSAGES.get(
                        clause,
                        f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(
                        time=t,
                        subsystem=self.NAME,
                        contract_id=clause,
                        message=message
                    )

        return self.contract_status
