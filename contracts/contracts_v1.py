# ship_in_transit_contracts.py
"""
Contract classes for the Ship-in-Transit architecture.

Pattern:
- Each block in the A-G diagram has its own contract class.
- Assumptions A* are checked first.
- Guarantees G* are evaluated only if all assumptions are True.
- Each class exposes an .evaluate() method returning a dict with
  the status of all A* and G* clauses (True / False / None).
"""


# contracts_runner.py

import numpy as np
import pandas as pd
import os
import csv

from typing import Dict, Optional, Sequence

from contracts.ship_dynamics import ShipDynamicsContract
from contracts.env_load import EnvironmentalLoadsContract
from contracts.machinery import MachineryContract
from contracts.rudder import RudderContract
from contracts.heading_controller import HeadingControllerContract
from contracts.throttle_controller import ThrottleControllerContract
from contracts.system import SystemLevelContract

######Improting
from env_wrappers.sea_env_ast_v2.env import SeaEnvASTv2



# ---------------------------------------------------------------------------
# Violation logger
# ---------------------------------------------------------------------------
class ViolationLogger:
    """
    Log contract violations in a human-friendly format:

    time, subsystem, contract_id, message

    Example:
    0.0,THRUST,A2,One or more thrusters are not working.
    """

    def __init__(self, csv_path: str, append: bool = False):
        self.csv_path = csv_path
        self.append = append
        self._ensure_header()

    def _ensure_header(self):
        file_exists = os.path.exists(self.csv_path)
        mode = "a" if (self.append and file_exists) else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if not file_exists or not self.append:
                writer.writerow(["time", "subsystem", "contract_id", "message"])

    def log(self, time: float, subsystem: str, contract_id: str, message: str):
        """
        Log a single violated clause.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time, subsystem, contract_id, message])



# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_finite_array(x: Optional[np.ndarray]) -> bool:
    if x is None:
        return False
    x = np.asarray(x)
    return x.size > 0 and np.all(np.isfinite(x))


# -------------------------------------------------------------
# Extract the LAST timestep from simulation_results
# -------------------------------------------------------------
def _latest_df_row(env):
    df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
    if df.empty:
        return None
    return df.iloc[-1]


# -------------------------------------------------------------
# Build contracts based on *one timestep only*
# -------------------------------------------------------------
def build_contracts_from_row(i, row, env: SeaEnvASTv2):
    # 1. Ship Dynamics ----------------------------------------
    thrust = np.array([[row["thrust force [kN]"] * 1000.0]])
    rudder_force = np.zeros_like(thrust)

    wave_n = row["wave force north [N]"]
    wave_e = row["wave force east [N]"]
    wind_n = row["wind force north [N]"]
    wind_e = row["wind force east [N]"]

    # environmental forces as 2D array (1 x 2)
    env_forces = np.array([[wave_n + wind_n,
                            wave_e + wind_e]])

    eta = np.array([[
        row["north position [m]"],
        row["east position [m]"],
        np.deg2rad(row["yaw angle [deg]"]),
    ]])

    nu = np.array([[
        row["forward speed [m/s]"],
        row["sideways speed [m/s]"],
        np.deg2rad(row["yaw rate [deg/sec]"]),
    ]])

    # per-timestep: we don’t really have history, so just zero accel
    acc = np.zeros_like(nu)

    

    ship_ct = ShipDynamicsContract(
        thrust_force=thrust,
        rudder_force=rudder_force,
        env_forces=env_forces,
        eta=eta,
        nu=nu,
        acc=acc,
        initial_state_valid=True,
        eom_residual=None,
    )

    # 2. Environmental Loads ----------------------------------
    params = {
        "wind_speed": row["wind speed [m/s]"],
        # "wind_dir_deg": row["wind dir [deg]"],
        "current_speed": row["current speed [m/s]"],
        # "current_dir_deg": row["current dir [deg]"],
        "significant_wave_height": row.get("significant wave height [m]", 0.0),
        "wave_period": row.get("wave period [s]", 0.0),
    }
    param_ranges = {
        "wind_speed": (0.0, 13.5), # 0 - 4 faulty
        # "wind_dir_deg": (-180.0, 180.0), # -180 t0 180 not
        "current_speed": (0.0, 1.0), #0 - 1.0 not
        # "current_dir_deg": (-180.0, 180.0), #not
        "significant_wave_height": (0.0, 2.5), # 0 - 5 not
        "wave_period": (0.0, 9.0), # 0 - 20 not
    }

    forces = np.array([[
        row["wave force north [N]"],
        row["wave force east [N]"],
        row["wave moment [Nm]"],
        row["wind force north [N]"],
        row["wind force east [N]"],
        row["wind moment [Nm]"],
    ]])

    env_ct = EnvironmentalLoadsContract(
        params=params,
        param_ranges=param_ranges,
        forces=forces,
        force_envelope={"max_norm": 2e6},
    )

    # 3. Machinery --------------------------------------------
    mach_ct = MachineryContract(
        shaft_speed=np.array([row["propeller shaft speed [rpm]"]]),
        mode_sequence=[0],
        mode_switch_times=[row["time [s]"]],
        dwell_time_min=1.0,
        thrust_force=thrust,
        power_demand=np.array([row["power [kw]"]]),
        power_available=np.array([
            row["available power me [kw]"]
            #+ row["available power electrical [kw]"] # we are not using electrical part yet
        ]),
        rate_limit_thrust=1e6,
        rate_limit_power=1e6,
    )

    # 4. Rudder -----------------------------------------------
    rudder_angle_rad = np.deg2rad(row["rudder angle [deg]"])
    rudder_ct = RudderContract(
        desired_angle=np.array([rudder_angle_rad]),
        actual_angle=np.array([rudder_angle_rad]),
        angle_limit=np.deg2rad(35.0),#35
        max_rate=np.deg2rad(2.0), #2.3
    )

    # 5. Heading ----------------------------------------------
    head_ct = HeadingControllerContract(
        waypoints_valid=True,
        waypoint_spacing_ok=True,
        speed_ok=True,
        rudder_ok=True,
        heading_ref=np.array([np.deg2rad(row["yaw angle [deg]"])]),
        cross_track_error=np.array([row["cross track error [m]"]]),
        max_cross_track=500.0, #cross track error
        reached_waypoint_flags=[True], #set it to None
    )


    # 6. Throttle ----------------------------------------------

    forward_speed = np.array([row.get("forward speed [m/s]", 0.0)]) 
    # shaft_speed_cmd = np.array([row["propeller shaft speed [rpm]"]])  
    shaft_speed_cmd = np.array([row.get("commanded load fraction me [-]", 0.0)])#change to the new shaft speed



    vmax = 6.0       # tune as needed
    tolerance = 0.5   # acceptable tracking error
    throttle_ct = ThrottleControllerContract(
        desired_speed=forward_speed,
        shaft_speed_cmd=shaft_speed_cmd,
        vmax=vmax,
        tolerance=tolerance,
    )

    # 7. System -----------------------------------------------
    sys_ct = SystemLevelContract(
        route_inside_chart=True,
        free_of_static_obstacles=True,
        sim_time=row["time [s]"],
        sim_time_max=52000.0,
        grounding_events=env.assets[0].ship_model.grounding_array[i],
        left_scenario_horizon=env.assets[0].ship_model.outside_horizon_array[i],
        navigation_success= not env.assets[0].ship_model.nav_failure_array[i],
        propulsion_overload_events=env.assets[0].ship_model.power_overload_array[i],
        travel_distance=0.0,  
        travel_time=row["time [s]"],
        distance_min=0.0,
        distance_max=1e9,
        time_min=0.0,
        time_max=52000.0,
    )

    return [ship_ct, env_ct, mach_ct, rudder_ct, head_ct, throttle_ct, sys_ct]



# -------------------------------------------------------------
# MAIN ENTRYPOINT — called *every timestep*
# -------------------------------------------------------------
# def evaluate_contracts_per_timestep(env, logger, run_id="run"):
#     row = _latest_df_row(env)
#     if row is None:
#         return

#     contracts = build_contracts_from_row(row, env)

#     t = row["time [s]"]
#     meta = {"run_id": run_id}

#     for ct in contracts:
#         ct.evaluate(logger=logger, t=t, meta=meta)


# -----------------------------------------------------------
# Evaluate all contracts for each row in the dataframe
# -----------------------------------------------------------
def evaluate_contracts_over_dataframe(df, 
                                      env:SeaEnvASTv2, 
                                      logger, 
                                      run_id):
    for i in range(len(df)):
        row = df.iloc[i]
        t = row["time [s]"]
        meta = {"run_id": run_id, "step": i}

        contracts = build_contracts_from_row(i, row, env)
        for ct in contracts:
            ct.evaluate(logger=logger, t=t, meta=meta)


    print(f"Logged contract evaluations for {len(df)} timesteps.")



# Columns = [time [s], 
#         north position [m],
#         east position [m], 
#         yaw angle [deg], 
#         rudder angle [deg], 
#         forward speed [m/s], 
#         sideways speed [m/s], 
#         yaw rate [deg/sec], 
#         propeller shaft speed [rpm], 
#         commanded load fraction me [-],
#         commanded load fraction hsg [-], 
#         power me [kw], 
#         available power me [kw], 
#         power electrical [kw], 
#         available power electrical [kw], 
#         power [kw], propulsion power [kw], 
#         fuel rate me [kg/s], 
#         fuel rate hsg [kg/s], 
#         fuel rate [kg/s], 
#         fuel consumption me [kg], 
#         fuel consumption hsg [kg], 
#         fuel consumption [kg], 
#         motor torque [Nm], 
#         thrust force [kN], 
#         cross track error [m], 
#         heading error [deg], 
#         wave force north [N], 
#         wave force east [N], 
#         wave moment [Nm], 
#         wind force north [N], 
#         wind force east [N], 
#         wind moment [Nm], 
#         wind speed [m/s], 
#         wind dir [deg], 
#         current speed [m/s], 
#         current dir [deg]]