# ---------------------------------------------------------------------------
# 6. Heading Controller Contract
# ---------------------------------------------------------------------------

from typing import Dict, Optional, Sequence
import numpy as np
import os
import csv
import json

class HeadingControllerContract:
    NAME = "HeadingController"
    MESSAGES = {
        "A1": "Waypoint list is invalid or spacing is too small.",
        "A2": "Ship speed or rudder authority insufficient for heading control.",
        "G1": "Cross-track error exceeds allowed limit.",
        "G2": "Not all waypoints were reached.",
    }


    def __init__(
        self,
        waypoints_valid: bool,
        waypoint_spacing_ok: bool,
        speed_ok: bool,
        rudder_ok: bool,
        heading_ref: np.ndarray,
        cross_track_error: np.ndarray,
        max_cross_track: float,
        reached_waypoint_flags: Sequence[bool],
    ):
        self.waypoints_valid = waypoints_valid
        self.waypoint_spacing_ok = waypoint_spacing_ok
        self.speed_ok = speed_ok
        self.rudder_ok = rudder_ok
        self.heading_ref = heading_ref
        self.cross_track_error = cross_track_error
        self.max_cross_track = max_cross_track
        self.reached_waypoint_flags = reached_waypoint_flags
        self.contract_status = {"A1": None, "A2": None, "G1": None, "G2": None}

    def check_A1(self):
        self.contract_status["A1"] = bool(self.waypoints_valid and self.waypoint_spacing_ok)
        return self.contract_status["A1"]

    def check_A2(self):
        self.contract_status["A2"] = bool(self.speed_ok and self.rudder_ok)
        return self.contract_status["A2"]

    def check_G1(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G1"] = None
        #     return None
        e = np.abs(np.asarray(self.cross_track_error))
        self.contract_status["G1"] = bool(np.all(e <= self.max_cross_track))
        return self.contract_status["G1"]

    def check_G2(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G2"] = None
        #     return None
        self.contract_status["G2"] = all(bool(f) for f in self.reached_waypoint_flags)
        return self.contract_status["G2"]

    def evaluate(self, logger=None, t=None, meta=None):
        self.check_A1()
        self.check_A2()
        self.check_G1()
        self.check_G2()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(t, self.NAME, clause, msg)

        return self.contract_status
