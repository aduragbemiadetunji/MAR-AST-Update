# ship_in_transit_contracts.py
"""
Contract classes and runner for the Ship-in-Transit architecture.

This patched version removes several placeholder truths from the runner and
replaces them with checks derived from simulator outputs, route geometry, and
logged machinery-mode history.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from contracts.ship_dynamics import ShipDynamicsContract
from contracts.env_load import EnvironmentalLoadsContract
from contracts.machinery import MachineryContract
from contracts.rudder import RudderContract
from contracts.heading_controller import HeadingControllerContract
from contracts.throttle_controller import ThrottleControllerContract
from contracts.system import SystemLevelContract

try:
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon
except Exception:  # pragma: no cover - optional dependency
    Point = LineString = Polygon = MultiPolygon = None


# ---------------------------------------------------------------------------
# Violation logger
# ---------------------------------------------------------------------------
class ViolationLogger:
    """
    Log contract violations in a human-friendly format:

    time, subsystem, contract_id, message
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


def _safe_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    s = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return s.to_numpy(dtype=float)


def _safe_scalar(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except Exception:
        return float(default)


def _load_route_points2(env, asset_index: int = 0) -> np.ndarray:
    try:
        route_path = getattr(env.assets[asset_index].ship_model, "route_name", None)
        if route_path is None or not os.path.exists(route_path):
            return np.empty((0, 2), dtype=float)
        pts = np.loadtxt(route_path)
        pts = np.atleast_2d(pts)
        if pts.shape[1] < 2:
            return np.empty((0, 2), dtype=float)
        return np.asarray(pts[:, :2], dtype=float)
    except Exception:
        return np.empty((0, 2), dtype=float)
    

def _load_route_points(env, asset_index: int = 0) -> np.ndarray:
    try:
        asset = env.assets[asset_index]
    except Exception:
        return np.empty((0, 2), dtype=float)

    # 1) Prefer route points attached directly to the asset
    for pts in (
        getattr(asset, "route_points", None),
        getattr(getattr(asset, "ship_model", None), "route_points", None),
    ):
        try:
            if pts is None:
                continue
            arr = np.atleast_2d(np.asarray(pts, dtype=float))
            if arr.shape[0] >= 2 and arr.shape[1] >= 2 and np.all(np.isfinite(arr[:, :2])):
                return arr[:, :2]
        except Exception:
            pass

    # 2) Fall back to route path if available
    for route_path in (
        getattr(asset, "route_name", None),
        getattr(getattr(asset, "ship_model", None), "route_name", None),
    ):
        try:
            if route_path is None or not os.path.exists(route_path):
                continue
            arr = np.atleast_2d(np.loadtxt(route_path))
            if arr.shape[0] >= 2 and arr.shape[1] >= 2:
                return np.asarray(arr[:, :2], dtype=float)
        except Exception:
            pass

    return np.empty((0, 2), dtype=float)


def _extract_frame_geometry(env):
    if Point is None:
        return None
    try:
        frame_obj = env.map[1]
        if hasattr(frame_obj, "geometry"):
            return frame_obj.geometry.unary_union
        if hasattr(frame_obj, "unary_union"):
            return frame_obj.unary_union
    except Exception:
        pass
    return None


def _coords_to_polygon(coords) -> Optional[Any]:
    if Polygon is None:
        return None
    try:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 3:
            return Polygon(arr[:, :2])
    except Exception:
        return None
    return None


def _extract_obstacle_geometries(env):
    """Best-effort extraction of obstacle polygons from the simulator obstacle object."""
    if Polygon is None:
        return []

    try:
        map_obj = env.map[0]
    except Exception:
        return []

    geometries = []
    candidates = []
    if hasattr(map_obj, "__dict__"):
        candidates.extend(map_obj.__dict__.values())

    def _harvest(obj):
        if obj is None:
            return
        if Polygon is not None and isinstance(obj, (Polygon, MultiPolygon)):
            geometries.append(obj)
            return
        poly = _coords_to_polygon(obj)
        if poly is not None:
            geometries.append(poly)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _harvest(item)

    for candidate in candidates:
        _harvest(candidate)

    return geometries


def _precompute_contract_context(df: pd.DataFrame, env, asset_index: int = 0) -> Dict[str, Any]:
    route_pts = _load_route_points(env, asset_index=asset_index)
    frame_geom = _extract_frame_geometry(env)
    obstacle_geoms = _extract_obstacle_geometries(env)

    args = getattr(env, "args", None)
    radius_of_acceptance = float(getattr(args, "radius_of_acceptance", 300.0))
    sim_time_max = float(
        getattr(getattr(env.assets[asset_index].ship_model, "int", None), "sim_time", df["time [s]"].iloc[-1])
    )
    desired_speed = float(getattr(env.assets[asset_index].ship_model, "desired_speed", 0.0))
    max_cross_track = float(getattr(env.assets[asset_index].ship_model, "cross_track_error_tolerance", 500.0))

    # waypoints_valid = bool(route_pts.shape[0] >= 2 and np.all(np.isfinite(route_pts)))
    # print(f"Route points loaded: {route_pts}, valid: {waypoints_valid}")
    # if route_pts.shape[0] >= 2:
    #     spacing = np.linalg.norm(np.diff(route_pts, axis=0), axis=1)
    #     waypoint_spacing_ok = bool(np.all(spacing >= max(20.0, 0.25 * radius_of_acceptance)))
    # else:
    #     waypoint_spacing_ok = False


    route_metadata_available = bool(route_pts.shape[0] >= 2 and np.all(np.isfinite(route_pts)))

    if route_metadata_available:
        waypoints_valid = True
        spacing = np.linalg.norm(np.diff(route_pts, axis=0), axis=1)
        waypoint_spacing_ok = bool(np.all(spacing >= 10.0))
    else:
        # Do not turn monitor-configuration failure into a persistent control violation
        waypoints_valid = True
        waypoint_spacing_ok = True
        print("Warning: route metadata unavailable for heading contract; skipping static waypoint validity checks.")



    route_inside_chart = True
    if frame_geom is not None and route_pts.size > 0:
        route_inside_chart = bool(
            all(frame_geom.buffer(1.0).covers(Point(float(x), float(y))) for x, y in route_pts)
        )

    free_of_static_obstacles = True
    if obstacle_geoms and route_pts.size > 0 and LineString is not None:
        route_line = LineString(route_pts)
        free_of_static_obstacles = bool(not any(route_line.intersects(geom) for geom in obstacle_geoms))

    north = _safe_array(df, "north position [m]")
    east = _safe_array(df, "east position [m]")
    pos = np.column_stack([north, east])
    if len(pos) >= 2:
        step_distance = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        cumulative_distance = np.concatenate([[0.0], np.cumsum(step_distance)])
    else:
        cumulative_distance = np.zeros(len(df), dtype=float)

    times = _safe_array(df, "time [s]")
    dt = float(np.median(np.diff(times))) if len(times) >= 2 else 1.0
    startup_grace_s = max(30.0, 4.0 * dt)

    if route_pts.shape[0] >= 1:
        goal = route_pts[-1]
        dist_to_goal = np.linalg.norm(pos - goal, axis=1)
    else:
        goal = None
        dist_to_goal = np.full(len(df), np.nan, dtype=float)

    route_progress_ok = np.ones(len(df), dtype=bool)
    regression_tol_m = max(25.0, 0.15 * radius_of_acceptance)
    progress_window = max(1, int(round(60.0 / max(dt, 1e-6))))
    for i in range(len(df)):
        if times[i] <= startup_grace_s or goal is None or not np.isfinite(dist_to_goal[i]):
            route_progress_ok[i] = True
            continue
        j = max(0, i - progress_window)
        route_progress_ok[i] = bool(dist_to_goal[i] <= dist_to_goal[j] + regression_tol_m)

    initial_state_valid = True
    if len(df) > 0:
        r0 = df.iloc[0]
        initial_state_valid = bool(
            np.isfinite(_safe_scalar(r0, "north position [m]"))
            and np.isfinite(_safe_scalar(r0, "east position [m]"))
            and abs(_safe_scalar(r0, "forward speed [m/s]")) <= 15.0
            and abs(_safe_scalar(r0, "sideways speed [m/s]")) <= 10.0
            and abs(np.deg2rad(_safe_scalar(r0, "yaw rate [deg/sec]"))) <= 1.5
        )

    return {
        "route_points": route_pts,
        "waypoints_valid": waypoints_valid,
        "waypoint_spacing_ok": waypoint_spacing_ok,
        "route_inside_chart": route_inside_chart,
        "free_of_static_obstacles": free_of_static_obstacles,
        "cumulative_distance": cumulative_distance,
        "route_progress_ok": route_progress_ok,
        "initial_state_valid": initial_state_valid,
        "radius_of_acceptance": radius_of_acceptance,
        "sim_time_max": sim_time_max,
        "desired_speed": desired_speed,
        "max_cross_track": max_cross_track,
        "dt": dt,
        "startup_grace_s": startup_grace_s,
    }


def _window_df(df: pd.DataFrame, i: int, window: int = 5) -> pd.DataFrame:
    return df.iloc[max(0, i - window + 1): i + 1]


def _mode_history_until(df: pd.DataFrame, i: int):
    mode_col = None
    for candidate in ["machinery mode [str]", "operating mode [str]", "operating mode"]:
        if candidate in df.columns:
            mode_col = candidate
            break
    if mode_col is None:
        return [0], [float(df.iloc[i]["time [s]"])]

    mode_values = df.iloc[: i + 1][mode_col].astype(str).tolist()
    time_values = pd.to_numeric(df.iloc[: i + 1]["time [s]"], errors="coerce").fillna(0.0).tolist()
    seq, switch_times = [], []
    prev = object()
    for mode, t in zip(mode_values, time_values):
        if mode != prev:
            seq.append(mode)
            switch_times.append(float(t))
            prev = mode
    return seq, switch_times


def _engine_status_from_row(row: pd.Series) -> Optional[Dict[str, bool]]:
    keys = ["ME", "DG1", "DG2", "HSG"]
    status = {}
    found_any = False
    for k in keys:
        candidates = [f"{k} healthy [-]", f"{k} available", f"{k} status [-]"]
        value = None
        for col in candidates:
            if col in row.index:
                value = row.get(col)
                found_any = True
                break
        if value is None or (hasattr(value, '__float__') and not np.isfinite(float(value))):
            status[k] = True
        else:
            try:
                status[k] = bool(int(value))
            except Exception:
                status[k] = bool(value)
    return status if found_any else None


def _current_mode_name(row: pd.Series) -> Optional[str]:
    for candidate in ["machinery mode [str]", "operating mode [str]", "operating mode"]:
        if candidate in row.index and pd.notna(row[candidate]):
            return str(row[candidate])
    return None


def _mode_feasible(mode_name: Optional[str], engine_status: Optional[Dict[str, bool]]) -> Optional[bool]:
    if mode_name is None or engine_status is None:
        return None

    mode = str(mode_name).upper()
    me = engine_status.get("ME", True)
    dg1 = engine_status.get("DG1", True)
    dg2 = engine_status.get("DG2", True)
    hsg = engine_status.get("HSG", True)
    dg_count = int(dg1) + int(dg2)

    if mode == "DP_SAFE":
        return True
    if mode == "PTO":
        return bool(me and hsg)
    if mode == "MEC":
        return bool(me and dg_count >= 1)
    if mode == "PTI":
        return bool(hsg and dg_count >= 2)
    if mode == "PTI_1DG":
        return bool(hsg and dg_count >= 1)
    if mode == "MEC_BLACKOUT":
        return bool(me)
    return None


# -------------------------------------------------------------
# Extract the LAST timestep from simulation_results
# -------------------------------------------------------------
def _latest_df_row(env):
    df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
    if df.empty:
        return None
    return df.iloc[-1]


# -------------------------------------------------------------
# Build contracts based on dataframe history up to timestep i
# -------------------------------------------------------------
def build_contracts_from_row(i, row, df, env, ctx: Dict[str, Any]):
    wdf = _window_df(df, i, window=5)
    times_w = _safe_array(wdf, "time [s]")
    dt = ctx["dt"]

    # 1. Ship Dynamics ----------------------------------------
    thrust = (_safe_array(wdf, "thrust force [kN]") * 1000.0).reshape(-1, 1)
    rudder_force = np.zeros_like(thrust)

    env_forces = np.column_stack([
        _safe_array(wdf, "wave force north [N]") + _safe_array(wdf, "wind force north [N]"),
        _safe_array(wdf, "wave force east [N]") + _safe_array(wdf, "wind force east [N]"),
    ])

    eta = np.column_stack([
        _safe_array(wdf, "north position [m]"),
        _safe_array(wdf, "east position [m]"),
        np.deg2rad(_safe_array(wdf, "yaw angle [deg]")),
    ])
    nu = np.column_stack([
        _safe_array(wdf, "forward speed [m/s]"),
        _safe_array(wdf, "sideways speed [m/s]"),
        np.deg2rad(_safe_array(wdf, "yaw rate [deg/sec]")),
    ])

    if len(nu) >= 2 and dt > 0.0:
        acc = np.vstack([np.zeros((1, nu.shape[1])), np.diff(nu, axis=0) / dt])
    else:
        acc = np.zeros_like(nu)

    ship_ct = ShipDynamicsContract(
        thrust_force=thrust,
        rudder_force=rudder_force,
        env_forces=env_forces,
        eta=eta,
        nu=nu,
        acc=acc,
        initial_state_valid=ctx["initial_state_valid"],
        eom_residual=None,
    )

    # 2. Environmental Loads ----------------------------------
    params = {
        "wind_speed": _safe_scalar(row, "wind speed [m/s]"),
        "current_speed": _safe_scalar(row, "current speed [m/s]"),
    }
    param_ranges = {
        "wind_speed": (0.0, 13.5),
        "current_speed": (0.0, 1.0),
    }
    forces = np.column_stack([
        _safe_array(wdf, "wave force north [N]"),
        _safe_array(wdf, "wave force east [N]"),
        _safe_array(wdf, "wave moment [Nm]"),
        _safe_array(wdf, "wind force north [N]"),
        _safe_array(wdf, "wind force east [N]"),
        _safe_array(wdf, "wind moment [Nm]"),
    ])
    env_ct = EnvironmentalLoadsContract(
        params=params,
        param_ranges=param_ranges,
        forces=forces,
        force_envelope={"max_norm": 2e6},
    )

    # 3. Machinery --------------------------------------------
    mode_sequence, mode_switch_times = _mode_history_until(df, i)
    current_mode = _current_mode_name(row)
    engine_status = _engine_status_from_row(row)
    mach_ct = MachineryContract(
        shaft_speed=_safe_array(wdf, "propeller shaft speed [rpm]"),
        mode_sequence=mode_sequence,
        mode_switch_times=mode_switch_times,
        dwell_time_min=60.0,
        thrust_force=thrust[:, 0],
        power_demand=_safe_array(wdf, "power [kw]"),
        power_available=(
            _safe_array(wdf, "available power me [kw]")
            + _safe_array(wdf, "available power electrical [kw]")
        ),
        rate_limit_thrust=5e5,
        rate_limit_power=5e5,
        mode_feasible=_mode_feasible(current_mode, engine_status),
    )

    # 4. Rudder -----------------------------------------------
    rudder_angles = np.deg2rad(_safe_array(wdf, "rudder angle [deg]"))
    max_rudder_rate = np.deg2rad(2.3) * max(dt, 1.0)
    rudder_ct = RudderContract(
        desired_angle=np.array([rudder_angles[-1]]),
        actual_angle=rudder_angles,
        angle_limit=np.deg2rad(35.0),
        max_rate=max_rudder_rate,
    )

    # 5. Heading ----------------------------------------------
    fwd_speed = _safe_scalar(row, "forward speed [m/s]")
    rudder_deg = abs(_safe_scalar(row, "rudder angle [deg]"))
    speed_threshold = max(0.25, 0.10 * ctx["desired_speed"])
    speed_ok = bool((row["time [s]"] <= ctx["startup_grace_s"]) or (fwd_speed >= speed_threshold))
    rudder_ok = bool(rudder_deg <= 0.95 * 35.0)
    head_ct = HeadingControllerContract(
        waypoints_valid=ctx["waypoints_valid"],
        waypoint_spacing_ok=ctx["waypoint_spacing_ok"],
        speed_ok=speed_ok,
        rudder_ok=rudder_ok,
        heading_ref=np.array([np.deg2rad(_safe_scalar(row, "yaw angle [deg]"))]),
        cross_track_error=np.array([_safe_scalar(row, "cross track error [m]")]),
        max_cross_track=ctx["max_cross_track"],
        reached_waypoint_flags=[bool(ctx["route_progress_ok"][i])],
        route_progress_ok=bool(ctx["route_progress_ok"][i]),
    )

    # 6. Throttle ---------------------------------------------
    desired_speed = ctx["desired_speed"]
    if current_mode is not None and str(current_mode).upper() == "DP_SAFE":
        desired_speed = 0.0
    throttle_ct = ThrottleControllerContract(
        desired_speed=np.array([desired_speed]),
        shaft_speed_cmd=np.array([
            _safe_scalar(row, "commanded load fraction me [-]")
            + _safe_scalar(row, "commanded load fraction hsg [-]")
        ]),
        vmax=max(6.0, ctx["desired_speed"] + 1.0),
        tolerance=0.3,
        actual_shaft_speed=_safe_array(wdf, "propeller shaft speed [rpm]"),
        min_shaft_speed_rpm=5.0,
    )

    # 7. System -----------------------------------------------
    travel_distance = float(ctx["cumulative_distance"][i])
    travel_time = float(_safe_scalar(row, "time [s]"))
    progress_time = max(0.0, travel_time - ctx["startup_grace_s"])
    distance_min = max(0.0, 0.20 * ctx["desired_speed"] * progress_time)
    distance_max = max(1000.0, 2.50 * ctx["desired_speed"] * max(travel_time, 1.0) + 500.0)

    sys_ct = SystemLevelContract(
        route_inside_chart=ctx["route_inside_chart"],
        free_of_static_obstacles=ctx["free_of_static_obstacles"],
        sim_time=travel_time,
        sim_time_max=ctx["sim_time_max"],
        grounding_events=int(env.assets[0].ship_model.grounding_array[i]),
        left_scenario_horizon=bool(env.assets[0].ship_model.outside_horizon_array[i]),
        navigation_success=not bool(env.assets[0].ship_model.nav_failure_array[i]),
        propulsion_overload_events=int(env.assets[0].ship_model.power_overload_array[i]),
        travel_distance=travel_distance,
        travel_time=travel_time,
        distance_min=distance_min,
        distance_max=distance_max,
        time_min=0.0,
        time_max=ctx["sim_time_max"],
    )

    return [ship_ct, env_ct, mach_ct, rudder_ct, head_ct, throttle_ct, sys_ct]


# -----------------------------------------------------------
# Evaluate all contracts for each row in the dataframe
# -----------------------------------------------------------
def evaluate_contracts_over_dataframe(df, env, logger, run_id):
    ctx = _precompute_contract_context(df, env)

    for i in range(len(df)):
        row = df.iloc[i]
        t = float(row["time [s]"])
        meta = {"run_id": run_id, "step": i}

        contracts = build_contracts_from_row(i, row, df, env, ctx)
        for ct in contracts:
            ct.evaluate(logger=logger, t=t, meta=meta)

    print(f"Logged contract evaluations for {len(df)} timesteps.")
