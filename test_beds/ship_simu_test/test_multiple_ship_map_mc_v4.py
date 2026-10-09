from pathlib import Path
import sys
import geopandas as gpd

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file -- SPAWNS AT DIFFERENT TIMES
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from env_wrappers.multiship_env.env import AssetInfo, ShipAsset, MultiShipEnv

from simulator.ship_in_transit.sub_systems.ship_model import ShipConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26, RudderConfiguration
from simulator.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, HeadingControllerGains
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration

## IMPORT FUNCTIONS
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf
from utils.animate_ais import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation_ais import plot_ship_status, plot_ship_and_real_map

from test_beds.ship_simu_test.mode_changer_helper import resilience_supervisor, apply_safe_state_overrides
from test_beds.ship_simu_test.monte_carlo.mc_faults import sample_fault_scenario
from contracts.contracts import evaluate_contracts_over_dataframe, ViolationLogger


### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


###############################################################################
# -----------------------
# DEFAULT USER EDIT SECTION
# -----------------------
OWN_SHIP_SPEC = {
    "route_filename": "Trondheim_Kristiansund_OS.txt",
    "name_tag": "Own ship",
    "desired_speed": 4.5,
    "print_status": True,
    "colav_mode": "sbmpc",
    "spawn_time_s": 0.0,
}

TARGET_SHIP_SPECS = [
    # {
    #     "route_filename": "Trondheim_Kristiansund_TS1.txt", "name_tag": "Target ship 1", "desired_speed": 3.8,
    #     "print_status": False,
    #     "colav_mode": None,
    #     "spawn_time_s": 800.00,
    # }, #useless
    {"route_filename": "Trondheim_Kristiansund_TS2.txt", "name_tag": "Target ship 1", "desired_speed": 3.5, "spawn_time_s": 6200.0},
    {"route_filename": "Trondheim_Kristiansund_TS3.txt", "name_tag": "Target ship 2", "desired_speed": 4.0, "spawn_time_s": 1800.0},
    {"route_filename": "Trondheim_Kristiansund_TS4.txt", "name_tag": "Target ship 3", "desired_speed": 4.0, "spawn_time_s": 15000.0},
    # {"route_filename": "Trondheim_Kristiansund_TS5.txt", "name_tag": "Target ship 5", "desired_speed": 4.0, "spawn_time_s": 20000.0}, #useless
    {"route_filename": "Trondheim_Kristiansund_TS6.txt", "name_tag": "Target ship 4", "desired_speed": 3.5, "spawn_time_s": 21500.0},
    # {"route_filename": "Trondheim_Kristiansund_TS7.txt", "name_tag": "Target ship 7", "desired_speed": 4.0, "spawn_time_s": 30000.0}, #useless
    {"route_filename": "Trondheim_Kristiansund_TS8.txt", "name_tag": "Target ship 5", "desired_speed": 4.0, "spawn_time_s": 500.0},
]

WEATHER_PROFILE = {
    "wave": {
        "minimum_wave_frequency": 0.4,
        "maximum_wave_frequency": 2.5,
        "wave_frequency_discrete_unit_count": 50,
        "minimum_spreading_angle": -np.pi,
        "maximum_spreading_angle": np.pi,
        "spreading_angle_discrete_unit_count": 10,
        "spreading_coefficient": 1,
        "rho": 1025.0,
    },
    "current": {
        "initial_current_velocity": 0.01,
        "current_velocity_standard_deviation": 0.0075,
        "current_velocity_decay_rate": 0.025,
        "initial_current_direction": np.deg2rad(0.0),
        "current_direction_standard_deviation": 0.025,
        "current_direction_decay_rate": 0.025,
    },
    "wind": {
        "initial_mean_wind_velocity": None,
        "mean_wind_velocity_decay_rate": 0.025,
        "mean_wind_velocity_standard_deviation": 0.005,
        "initial_wind_direction": np.deg2rad(0.0),
        "wind_direction_decay_rate": 0.025,
        "wind_direction_standard_deviation": 0.025,
        "minimum_mean_wind_velocity": 0.0,
        "maximum_mean_wind_velocity": 42.0,
        "minimum_wind_gust_frequency": 0.06,
        "maximum_wind_gust_frequency": 0.4,
        "wind_gust_frequency_discrete_unit_count": 100,
        "clip_speed_nonnegative": True,
        "kappa_parameter": 0.0026,
        "U10": 10.0,
        "wind_evaluation_height": 5.0,
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ship in Transit Multi-Ship Simulation")

    parser.add_argument("--time_step", type=int, default=10, metavar="TIMESTEP",
                        help="ENV: time step size in second for ship transit simulator (default: 5)")
    parser.add_argument("--engine_step_count", type=int, default=10, metavar="ENGINE_STEP_COUNT",
                        help="ENV: engine integration step count in between simulation timestep (default: 10)")
    parser.add_argument("--radius_of_acceptance", type=int, default=300, metavar="ROA",
                        help="ENV: radius of acceptance in meter for LOS algorithm (default: 300)")
    parser.add_argument("--lookahead_distance", type=int, default=1000, metavar="LD",
                        help="ENV: lookahead distance in meter for LOS algorithm (default: 1000)")
    parser.add_argument("--nav_fail_time", type=int, default=300, metavar="NAV_FAIL_TIME",
                        help="ENV: Allowed recovery time in second from navigational failure warning condition (default: 300)")
    parser.add_argument("--ship_draw", type=bool, default=True, metavar="SHIP_DRAW",
                        help="ENV: record ship drawing for plotting and animation (default: True)")
    parser.add_argument("--time_since_last_ship_drawing", default=30, metavar="SHIP_DRAW_TIME",
                        help="ENV: time delay in second between ship drawing record (default: 30)")

    parser.add_argument("--n_episodes", type=int, default=1, metavar="N_EPISODES",
                        help="MC: number of simulation episodes (default: 1)")
    parser.add_argument("--base_seed", type=int, default=10, metavar="BASE_SEED",
                        help="MC: base seed used to generate reproducible episode seeds (default: 10)")
    parser.add_argument("--simulation_time", type=float, default=50000.0, metavar="SIM_TIME",
                        help="Simulation time in seconds for all assets")
    parser.add_argument("--output_dir", type=str, default="multi_ship_mc_outputs", metavar="OUTPUT_DIR",
                        help="Directory for summary csv and optional per-episode csv files")
    parser.add_argument("--plot_results", action="store_true", default=True,
                        help="Plot/animate the last episode. For campaigns, leave this off.")
    parser.add_argument("--save_episode_results", action="store_true", default=False,
                        help="Save one csv per asset for every episode.")
    parser.add_argument("--n_fault_events", type=int, default=1,
                        help="Number of own-ship fault events to sample per episode (default: 4)")
    parser.add_argument('--enable_fallback', action='store_true', default=False, 
                        help='Enable fallback behavior when primary control mode fails')
    return parser


GPKG_PATH = get_map_path(ROOT, "Trondheim_Kristiansund.gpkg")
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER = "land_3857"
COAST_LAYER = "coast_3857"
WATER_LAYER = None#"water_3857"
FERRY_ROUTE_LAYER = None


def load_map_data():
    frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = get_gdf_from_gpkg(
        GPKG_PATH,
        FRAME_LAYER,
        OCEAN_LAYER,
        LAND_LAYER,
        COAST_LAYER,
        WATER_LAYER,
        FERRY_ROUTE_LAYER,
    )
    map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf
    map_data = get_polygon_from_gdf(land_gdf)
    map_obj = [PolygonObstacle(map_data), frame_gdf]
    return map_obj, map_gdfs


def build_ship_configuration() -> ShipConfiguration:
    return ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=200000,
        ballast=200000,
        length_of_ship=80,
        width_of_ship=16,
        added_mass_coefficient_in_surge=0.4,
        added_mass_coefficient_in_sway=0.4,
        added_mass_coefficient_in_yaw=0.4,
        dead_weight_tonnage=3850000,
        mass_over_linear_friction_coefficient_in_surge=130,
        mass_over_linear_friction_coefficient_in_sway=18,
        mass_over_linear_friction_coefficient_in_yaw=90,
        nonlinear_friction_coefficient__in_surge=2400,
        nonlinear_friction_coefficient__in_sway=4000,
        nonlinear_friction_coefficient__in_yaw=400,
    )


MAIN_ENGINE_CAPACITY = 2160e3
DIESEL_GEN_CAPACITY = 510e3
HYBRID_SHAFT_GEN_AS_GENERATOR = "GEN"
HYBRID_SHAFT_GEN_AS_MOTOR = "MOTOR"
HYBRID_SHAFT_GEN_AS_OFFLINE = "OFF"


def build_wave_model_config(args) -> WaveModelConfiguration:
    cfg = WEATHER_PROFILE["wave"]
    return WaveModelConfiguration(
        minimum_wave_frequency=cfg["minimum_wave_frequency"],
        maximum_wave_frequency=cfg["maximum_wave_frequency"],
        wave_frequency_discrete_unit_count=cfg["wave_frequency_discrete_unit_count"],
        minimum_spreading_angle=cfg["minimum_spreading_angle"],
        maximum_spreading_angle=cfg["maximum_spreading_angle"],
        spreading_angle_discrete_unit_count=cfg["spreading_angle_discrete_unit_count"],
        spreading_coefficient=cfg["spreading_coefficient"],
        rho=cfg["rho"],
        timestep_size=args.time_step,
    )


def build_current_model_config(args) -> CurrentModelConfiguration:
    cfg = WEATHER_PROFILE["current"]
    return CurrentModelConfiguration(
        initial_current_velocity=cfg["initial_current_velocity"],
        current_velocity_standard_deviation=cfg["current_velocity_standard_deviation"],
        current_velocity_decay_rate=cfg["current_velocity_decay_rate"],
        initial_current_direction=cfg["initial_current_direction"],
        current_direction_standard_deviation=cfg["current_direction_standard_deviation"],
        current_direction_decay_rate=cfg["current_direction_decay_rate"],
        timestep_size=args.time_step,
    )


def build_wind_model_config(args) -> WindModelConfiguration:
    cfg = WEATHER_PROFILE["wind"]
    return WindModelConfiguration(
        initial_mean_wind_velocity=cfg["initial_mean_wind_velocity"],
        mean_wind_velocity_decay_rate=cfg["mean_wind_velocity_decay_rate"],
        mean_wind_velocity_standard_deviation=cfg["mean_wind_velocity_standard_deviation"],
        initial_wind_direction=cfg["initial_wind_direction"],
        wind_direction_decay_rate=cfg["wind_direction_decay_rate"],
        wind_direction_standard_deviation=cfg["wind_direction_standard_deviation"],
        minimum_mean_wind_velocity=cfg["minimum_mean_wind_velocity"],
        maximum_mean_wind_velocity=cfg["maximum_mean_wind_velocity"],
        minimum_wind_gust_frequency=cfg["minimum_wind_gust_frequency"],
        maximum_wind_gust_frequency=cfg["maximum_wind_gust_frequency"],
        wind_gust_frequency_discrete_unit_count=cfg["wind_gust_frequency_discrete_unit_count"],
        clip_speed_nonnegative=cfg["clip_speed_nonnegative"],
        kappa_parameter=cfg["kappa_parameter"],
        U10=cfg["U10"],
        wind_evaluation_height=cfg["wind_evaluation_height"],
        timestep_size=args.time_step,
    )


def build_machinery_config() -> MachinerySystemConfiguration:
    pto_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=MAIN_ENGINE_CAPACITY,
        electrical_capacity=0,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_GENERATOR,
        name_tag="PTO",
    ))
    mec_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=MAIN_ENGINE_CAPACITY,
        electrical_capacity=DIESEL_GEN_CAPACITY,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_OFFLINE,
        name_tag="MEC",
    ))
    pti_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=0,
        electrical_capacity=2 * DIESEL_GEN_CAPACITY,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_MOTOR,
        name_tag="PTI",
    ))
    pti_1dg_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=0,
        electrical_capacity=1 * DIESEL_GEN_CAPACITY,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_MOTOR,
        name_tag="PTI_1DG",
    ))
    mec_blackout_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=MAIN_ENGINE_CAPACITY,
        electrical_capacity=0,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_OFFLINE,
        name_tag="MEC_BLACKOUT",
    ))
    dp_safe_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=0,
        electrical_capacity=0,
        shaft_generator_state=HYBRID_SHAFT_GEN_AS_OFFLINE,
        name_tag="DP_SAFE",
    ))

    mso_modes = MachineryModes([
        pto_mode,
        mec_mode,
        pti_mode,
        pti_1dg_mode,
        mec_blackout_mode,
        dp_safe_mode,
    ])

    fuel_spec_me = SpecificFuelConsumptionWartila6L26()
    fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()

    return MachinerySystemConfiguration(
        machinery_modes=mso_modes,
        machinery_operating_mode=1,
        linear_friction_main_engine=68,
        linear_friction_hybrid_shaft_generator=57,
        gear_ratio_between_main_engine_and_propeller=0.6,
        gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
        propeller_inertia=6000,
        propeller_diameter=3.1,
        propeller_speed_to_torque_coefficient=7.5,
        propeller_speed_to_thrust_force_coefficient=1.7,
        hotel_load=200000,
        rated_speed_main_engine_rpm=1000,
        rudder_angle_to_sway_force_coefficient=50e3,
        rudder_angle_to_yaw_force_coefficient=500e3,
        max_rudder_angle_degrees=35,
        max_rudder_rate_degree_per_s=2.3,
        specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
        specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients(),
    )


def default_throttle_gains() -> ThrottleControllerGains:
    return ThrottleControllerGains(
        kp_ship_speed=2.50,
        ki_ship_speed=0.025,
        kp_shaft_speed=0.05,
        ki_shaft_speed=0.0001,
    )


def default_heading_gains() -> HeadingControllerGains:
    return HeadingControllerGains(kp=1.5, kd=75, ki=0.001)


def build_los_parameters(args) -> LosParameters:
    return LosParameters(
        radius_of_acceptance=args.radius_of_acceptance,
        lookahead_distance=args.lookahead_distance,
        integral_gain=0.002,
        integrator_windup_limit=4000,
    )


def load_route_points(route_name: str) -> np.ndarray:
    return np.loadtxt(route_name)


def infer_initial_yaw_deg(route_points: np.ndarray, fallback_deg: float = 0.0) -> float:
    if len(route_points) < 2:
        return fallback_deg
    dx = route_points[1, 0] - route_points[0, 0]
    dy = route_points[1, 1] - route_points[0, 1]
    if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
        return fallback_deg
    return float(np.rad2deg(np.arctan2(dy, dx)))


def build_simulation_configuration(route_name: str, args, initial_yaw_deg: float = None) -> SimulationConfiguration:
    route_points = load_route_points(route_name)
    start_east, start_north = route_points[0]
    yaw_deg = infer_initial_yaw_deg(route_points) if initial_yaw_deg is None else initial_yaw_deg
    return SimulationConfiguration(
        initial_north_position_m=start_east,
        initial_east_position_m=start_north,
        initial_yaw_angle_rad=np.deg2rad(yaw_deg),
        initial_forward_speed_m_per_s=0.0,
        initial_sideways_speed_m_per_s=0.0,
        initial_yaw_rate_rad_per_s=0.0,
        integration_step=args.time_step,
        simulation_time=args.simulation_time,
    )


def build_ship_asset(spec: dict,
                     args,
                     ship_config: ShipConfiguration,
                     machinery_config: MachinerySystemConfiguration,
                     wave_model_config: WaveModelConfiguration,
                     current_model_config: CurrentModelConfiguration,
                     wind_model_config: WindModelConfiguration,
                     map_obj,
                     is_own_ship: bool = False) -> ShipAsset:
    route_name = get_ship_route_path(ROOT, spec["route_filename"])
    simulation_config = build_simulation_configuration(
        route_name,
        args,
        initial_yaw_deg=spec.get("initial_yaw_deg", None),
    )

    nominal_desired_speed = spec.get("desired_speed", 4.0)
    spawn_time_s = float(spec.get("spawn_time_s", 0.0))
    initial_desired_speed = nominal_desired_speed if spawn_time_s <= 0.0 or is_own_ship else 0.0

    ship_model = ShipModel(
        ship_config=ship_config,
        simulation_config=simulation_config,
        wave_model_config=wave_model_config,
        current_model_config=current_model_config,
        wind_model_config=wind_model_config,
        machinery_config=machinery_config,
        throttle_controller_gain=spec.get("throttle_controller_gains", default_throttle_gains()),
        heading_controller_gain=spec.get("heading_controller_gains", default_heading_gains()),
        los_parameters=spec.get("los_parameters", build_los_parameters(args)),
        name_tag=spec["name_tag"],
        route_name=route_name,
        engine_steps_per_time_step=args.engine_step_count,
        initial_propeller_shaft_speed_rad_per_s=spec.get("initial_propeller_shaft_speed_rpm", 0.0) * np.pi / 30.0,
        initial_propeller_shaft_acc_rad_per_sec2=spec.get("initial_propeller_shaft_acceleration_rpm_per_s", 0.0) * np.pi / 30.0,
        desired_speed=initial_desired_speed,
        cross_track_error_tolerance=spec.get("cross_track_error_tolerance", 750),
        nav_fail_time=args.nav_fail_time,
        map_obj=map_obj[0],
        colav_mode=spec.get("colav_mode", "sbmpc"),
        print_status=spec.get("print_status", False),
    )

    info = AssetInfo(
        current_north=ship_model.north,
        current_east=ship_model.east,
        current_yaw_angle=ship_model.yaw_angle,
        forward_speed=ship_model.forward_speed,
        sideways_speed=ship_model.sideways_speed,
        name_tag=ship_model.name_tag,
        ship_length=ship_model.l_ship,
        ship_width=ship_model.w_ship,
    )
    asset = ShipAsset(ship_model=ship_model, info=info)
    asset.spawn_time_s = spawn_time_s
    asset.nominal_desired_speed = nominal_desired_speed
    asset.is_spawned = bool(is_own_ship or spawn_time_s <= 0.0)
    asset.is_own_ship = bool(is_own_ship)
    asset.route_name = route_name
    asset.route_points = np.atleast_2d(load_route_points(route_name))[:, :2]
    return asset


def build_assets(args, map_obj) -> List[ShipAsset]:
    ship_config = build_ship_configuration()
    machinery_config = build_machinery_config()
    wave_model_config = build_wave_model_config(args)
    current_model_config = build_current_model_config(args)
    wind_model_config = build_wind_model_config(args)

    own_ship_asset = build_ship_asset(
        OWN_SHIP_SPEC,
        args,
        ship_config,
        machinery_config,
        wave_model_config,
        current_model_config,
        wind_model_config,
        map_obj,
        is_own_ship=True,
    )

    target_assets = []
    for idx, spec in enumerate(TARGET_SHIP_SPECS, start=1):
        if "name_tag" not in spec:
            spec = dict(spec)
            spec["name_tag"] = f"Target ship {idx}"
        target_assets.append(
            build_ship_asset(
                spec,
                args,
                ship_config,
                machinery_config,
                wave_model_config,
                current_model_config,
                wind_model_config,
                map_obj,
                is_own_ship=False,
            )
        )

    return [own_ship_asset] + target_assets


def build_environment(args, assets, map_obj) -> MultiShipEnv:
    return MultiShipEnv(
        assets=assets,
        map=map_obj,
        wave_model_config=build_wave_model_config(args),
        current_model_config=build_current_model_config(args),
        wind_model_config=build_wind_model_config(args),
        args=args,
        include_wave=True,
        include_wind=False,
        include_current=False,
    )


def safe_last(results_dict: dict, key: str, default=0.0):
    values = results_dict.get(key, [])
    if values is None or len(values) == 0:
        return default
    return values[-1]


def series_abs_max(results_dict: dict, key: str, default=0.0):
    values = results_dict.get(key, [])
    if values is None or len(values) == 0:
        return default
    return float(np.max(np.abs(np.asarray(values))))


def compute_distance_travelled(results_df: pd.DataFrame) -> float:
    north_col = None
    east_col = None
    for candidate in ["north position [m]", "north position [m]"]:
        if candidate in results_df.columns:
            north_col = candidate
            break
    for candidate in ["east position [m]", "east position [m]"]:
        if candidate in results_df.columns:
            east_col = candidate
            break
    if north_col is None or east_col is None or len(results_df) < 2:
        return 0.0
    dn = results_df[north_col].diff().fillna(0.0)
    de = results_df[east_col].diff().fillna(0.0)
    return float(np.sqrt(dn**2 + de**2).sum())


def summarise_asset_result(asset: ShipAsset) -> dict:
    results_df = pd.DataFrame.from_dict(asset.ship_model.simulation_results)
    summary = {
        "final_time_s": float(asset.ship_model.int.time),
        "distance_travelled_m": compute_distance_travelled(results_df),
        "fuel_consumption_kg": float(safe_last(asset.ship_model.simulation_results, "fuel consumption [kg]", 0.0)),
        "max_cross_track_error_m": float(series_abs_max(asset.ship_model.simulation_results, "cross track error [m]", 0.0)),
        "max_speed_m_per_s": float(series_abs_max(asset.ship_model.simulation_results, "forward speed [m/s]", 0.0)),
        "max_thrust_kN": float(series_abs_max(asset.ship_model.simulation_results, "thrust force [kN]", 0.0)),
    }
    return summary


def summarize_violations(viol_df: pd.DataFrame) -> dict:
    summary = {}
    if viol_df is None or viol_df.empty:
        summary["total_contract_violations"] = 0
        return summary

    summary["total_contract_violations"] = int(len(viol_df))

    if {"subsystem", "contract_id"}.issubset(viol_df.columns):
        grouped = viol_df.groupby(["subsystem", "contract_id"]).size()
        for (subsystem, contract_id), count in grouped.items():
            summary[f"viol_{subsystem}_{contract_id}"] = int(count)

    return summary


def sample_multi_fault_schedule(rng: np.random.Generator, sim_time_s: float, n_fault_events: int):
    events = []
    if n_fault_events <= 0:
        return events

    start_min = 0.05 * sim_time_s
    start_max = 0.80 * sim_time_s
    gap = max(sim_time_s / max(n_fault_events + 1, 2), 1.0)

    for i in range(n_fault_events):
        local_start_min = start_min + i * gap * 0.85
        local_start_max = min(start_max, local_start_min + gap)
        if local_start_max <= local_start_min:
            local_start_max = local_start_min + 1.0
        event = sample_fault_scenario(rng, sim_time_s=sim_time_s, start_frac=(local_start_min / sim_time_s, local_start_max / sim_time_s), duration_range_s=(120.0, 600.0))
        events.append(event)

    events.sort(key=lambda e: e.start_s)
    return events


def aggregate_engine_status(fault_events, running_time: float):
    engine_status = {"ME": True, "DG1": True, "DG2": True, "HSG": True}
    active_labels = []
    active_events = []
    for idx, event in enumerate(fault_events, start=1):
        if event.active(running_time):
            active_labels.append(f"F{idx}")
            active_events.append(event)
            event_status = event.engine_status(running_time)
            for eng in engine_status:
                engine_status[eng] = engine_status[eng] and bool(event_status[eng])
    return engine_status, active_labels, active_events


def build_logged_own_ship_results_df(asset: ShipAsset) -> pd.DataFrame:
    result_df = pd.DataFrame.from_dict(asset.ship_model.simulation_results)
    n = len(result_df)

    histories = getattr(asset, "logged_histories", None)
    if not histories:
        return result_df

    result_df["operating mode"] = histories["mode_history"][:n]
    result_df["fault active"] = histories["fault_active_history"][:n]
    result_df["fault start [s]"] = histories["fault_start_history"][:n]
    result_df["fault end [s]"] = histories["fault_end_history"][:n]

    result_df["ME available"] = histories["me_status_history"][:n]
    result_df["DG1 available"] = histories["dg1_status_history"][:n]
    result_df["DG2 available"] = histories["dg2_status_history"][:n]
    result_df["HSG available"] = histories["hsg_status_history"][:n]

    result_df["decision action"] = histories["decision_action_history"][:n]
    result_df["decision reason"] = histories["decision_reason_history"][:n]
    result_df["active fault count"] = histories["active_fault_count_history"][:n]

    result_df["available engine count"] = (
        result_df["ME available"].astype(int)
        + result_df["DG1 available"].astype(int)
        + result_df["DG2 available"].astype(int)
        + result_df["HSG available"].astype(int)
    )

    result_df["significant wave height [m]"] = histories["significant_wave_height"][:n]
    result_df["wave period [s]"] = histories["wave_period"][:n]

    return result_df


def update_spawned_assets(assets: List[ShipAsset], running_time: float):
    newly_spawned = []
    for asset in assets[1:]:
        if (not getattr(asset, "is_spawned", True)) and running_time >= getattr(asset, "spawn_time_s", 0.0):
            asset.is_spawned = True
            asset.ship_model.desired_speed = getattr(asset, "nominal_desired_speed", asset.ship_model.desired_speed)
            newly_spawned.append(asset.ship_model.name_tag)
        elif not getattr(asset, "is_spawned", True):
            asset.ship_model.desired_speed = 0.0
    return newly_spawned


def propulsion_fraction_from_mode(current_mode, engine_status):
    mode = str(current_mode).upper()

    me_ok  = int(engine_status.get("ME", True))
    dg_ok  = int(engine_status.get("DG1", True)) + int(engine_status.get("DG2", True))
    hsg_ok = int(engine_status.get("HSG", True))

    # ME-driven modes
    if mode in ("PTO", "MEC", "MEC_BLACKOUT"):
        return 1.0 if me_ok else 0.0

    # Electrical propulsion modes
    if mode == "PTI":
        if not hsg_ok:
            return 0.0
        return dg_ok / 2.0   # 0, 0.5, or 1.0

    if mode == "PTI_1DG":
        return 1.0 if (hsg_ok and dg_ok >= 1) else 0.0

    if mode == "DP_SAFE":
        return 0.0

    return 1.0


def run_episode(args, episode_idx: int, episode_seed: int, map_obj, map_gdfs):
    np.random.seed(episode_seed)
    assets = build_assets(args, map_obj)
    env = build_environment(args, assets, map_obj)
    env.reset()

    own_ship = assets[0].ship_model
    mach = own_ship.ship_machinery_model
    last_mode_change_time = 0.0
    min_dwell_s = 60.0
    mode_switch_count = 0
    min_available_engine_count = 4
    hotel_load_w = build_machinery_config().hotel_load

    scenario_rng = np.random.default_rng(episode_seed)
    fault_events = sample_multi_fault_schedule(
        scenario_rng,
        sim_time_s=own_ship.int.sim_time,
        n_fault_events=args.n_fault_events,
    )

    print(f"[EP {episode_idx:03d}] seed={episode_seed} n_fault_events={len(fault_events)}")
    for idx, event in enumerate(fault_events, start=1):
        print(f"    F{idx}: start={event.start_s:.1f}s dur={event.duration_s:.1f}s mask={event.fault_mask}")

    mode_history = []
    fault_active_history = []
    fault_start_history = []
    fault_end_history = []
    me_status_history = []
    dg1_status_history = []
    dg2_status_history = []
    hsg_status_history = []
    decision_action_history = []
    decision_reason_history = []
    active_fault_count_history = []
    significant_wave_height = []
    wave_period = []

    nominal_desired_speed = float(assets[0].ship_model.desired_speed)
    desired_speed_history = []
    propulsion_fraction_history = []

    running_time = 0.0
    while running_time < own_ship.int.sim_time and env.stop is False:
        spawned_now = update_spawned_assets(assets, running_time)
        for name in spawned_now:
            print(f"[EP {episode_idx:03d} | t={running_time:.1f}s] SPAWN: {name}")

        engine_status, active_faults, active_events = aggregate_engine_status(fault_events, running_time)
        min_available_engine_count = min(min_available_engine_count, int(sum(engine_status.values())))
        prop_frac = propulsion_fraction_from_mode(mach.operating_mode, engine_status)


        fault_active = len(active_events) > 0
        active_fault_count = len(active_events)
        if fault_active:
            fault_start_s = min(ev.start_s for ev in active_events)
            fault_end_s = max(getattr(ev, "end_s", ev.start_s + getattr(ev, "duration_s", 0.0)) for ev in active_events)
        else:
            fault_start_s = np.nan
            fault_end_s = np.nan

        # if running_time >= 10000 and running_time < 12000:
        #     env.Hs = 2.5
        #     env.Tp = 9.0
        #     env.psi_0 = np.deg2rad(0.0)
            
        #     # Current
        #     env.vel_mean = 1.0
        #     env.current_dir_mean = np.deg2rad(56.0)
            
        #     # Wind
        #     env.Ubar_mean = 15
        #     env.wind_dir_mean = np.deg2rad(90.0)

        env.step()

        required_hotel_power_w = hotel_load_w
        required_propulsion_power_w = 1000.0 * safe_last(
            own_ship.simulation_results,
            "propulsion power [kw]",
            0.0,
        )
        time_in_mode = running_time - last_mode_change_time

        decision = resilience_supervisor(
            current_mode=mach.operating_mode,
            engine_status=engine_status,
            required_propulsion_power=required_propulsion_power_w,
            required_hotel_power=required_hotel_power_w,
            me_capacity=MAIN_ENGINE_CAPACITY,
            dg_capacity=DIESEL_GEN_CAPACITY,
            min_dwell_s=min_dwell_s,
            time_in_mode_s=time_in_mode,
        )


        if not args.enable_fallback:
            # No resilience manager: keep the same mode, but degrade propulsion
            own_ship.desired_speed = nominal_desired_speed * prop_frac

        else:
            own_ship.desired_speed = nominal_desired_speed
            if decision.action in ("SWITCH_MODE", "ENTER_SAFE_STATE"):
                if mach.operating_mode != decision.recommended_mode:
                    print(
                        f"[EP {episode_idx:03d} | t={running_time:.1f}s] "
                        f"{decision.action}: {mach.operating_mode} -> {decision.recommended_mode} | {decision.reason}"
                    )
                    mach.operating_mode = decision.recommended_mode
                    last_mode_change_time = running_time
                    mode_switch_count += 1

            apply_safe_state_overrides(own_ship)

        mode_history.append(str(mach.operating_mode))
        fault_active_history.append(bool(fault_active))
        fault_start_history.append(fault_start_s)
        fault_end_history.append(fault_end_s)
        me_status_history.append(int(engine_status.get("ME", True)))
        dg1_status_history.append(int(engine_status.get("DG1", True)))
        dg2_status_history.append(int(engine_status.get("DG2", True)))
        hsg_status_history.append(int(engine_status.get("HSG", True)))
        decision_action_history.append(str(decision.action))
        decision_reason_history.append(str(decision.reason))
        active_fault_count_history.append(int(active_fault_count))
        significant_wave_height.append(env.Hs)
        wave_period.append(env.Tp)

        desired_speed_history.append(float(own_ship.desired_speed))
        propulsion_fraction_history.append(float(prop_frac))


        running_time = np.max([asset.ship_model.int.time for asset in assets])

    assets[0].logged_histories = {
        "mode_history": mode_history,
        "fault_active_history": fault_active_history,
        "fault_start_history": fault_start_history,
        "fault_end_history": fault_end_history,
        "me_status_history": me_status_history,
        "dg1_status_history": dg1_status_history,
        "dg2_status_history": dg2_status_history,
        "hsg_status_history": hsg_status_history,
        "decision_action_history": decision_action_history,
        "decision_reason_history": decision_reason_history,
        "active_fault_count_history": active_fault_count_history,
        "significant_wave_height": significant_wave_height,
        "wave_period": wave_period,

        "desired_speed_history": desired_speed_history,
        "propulsion_fraction_history": propulsion_fraction_history,
    }
    assets[0].logged_results_df = build_logged_own_ship_results_df(assets[0])

    # Add contract-monitor aliases so the runner can use physically meaningful
    # logged states without breaking existing analysis/plotting code.
    assets[0].logged_results_df["operating mode [str]"] = assets[0].logged_results_df["operating mode"].astype(str)
    assets[0].logged_results_df["ME healthy [-]"] = assets[0].logged_results_df["ME available"].astype(int)
    assets[0].logged_results_df["DG1 healthy [-]"] = assets[0].logged_results_df["DG1 available"].astype(int)
    assets[0].logged_results_df["DG2 healthy [-]"] = assets[0].logged_results_df["DG2 available"].astype(int)
    assets[0].logged_results_df["HSG healthy [-]"] = assets[0].logged_results_df["HSG available"].astype(int)

    # Contract evaluation for the own ship only
    os.makedirs(args.output_dir, exist_ok=True)
    viol_csv_path = os.path.join(
        args.output_dir,
        f"violations_episode_{episode_idx:03d}_seed_{episode_seed}.csv"
    )
    logger = ViolationLogger(viol_csv_path, append=False)

    evaluate_contracts_over_dataframe(
        assets[0].logged_results_df,
        env,
        logger,
        run_id=f"episode_{episode_idx:03d}_seed_{episode_seed}"
    )

    try:
        viol_df = pd.read_csv(viol_csv_path)
    except Exception:
        viol_df = pd.DataFrame()

    mode_series = assets[0].logged_results_df["operating mode"].astype(str)
    mode_switch_count = int((mode_series != mode_series.shift(1)).sum() - 1)
    dp_safe_entries = int(((mode_series == "DP_SAFE") & (mode_series.shift(1) != "DP_SAFE")).sum())

    episode_row = {
        "episode": episode_idx,
        "seed": episode_seed,
        "n_fault_events": int(len(fault_events)),
        "fault_windows": str([
            {"start_s": float(e.start_s), "duration_s": float(e.duration_s), "fault_mask": e.fault_mask}
            for e in fault_events
        ]),
        "min_available_engine_count": int(min_available_engine_count),
        "mode_switch_count": int(mode_switch_count),
        "final_mode": str(mach.operating_mode),
        "env_stop": bool(env.stop),
        "dp_safe_entries": dp_safe_entries,
        "sim_completion_fraction": float(own_ship.int.time / own_ship.int.sim_time) if own_ship.int.sim_time > 0 else 0.0,
    }

    for asset_idx, asset in enumerate(assets):
        prefix = "own_ship" if asset_idx == 0 else f"target_ship_{asset_idx}"
        asset_summary = summarise_asset_result(asset)
        for key, value in asset_summary.items():
            episode_row[f"{prefix}_{key}"] = value
        if asset_idx > 0:
            episode_row[f"{prefix}_spawn_time_s"] = float(getattr(asset, "spawn_time_s", 0.0))

    episode_row.update(summarize_violations(viol_df))

    return episode_row, assets, env, map_gdfs


def save_episode_csvs(args, episode_idx: int, assets: List[ShipAsset]):
    os.makedirs(args.output_dir, exist_ok=True)
    for asset_idx, asset in enumerate(assets):
        asset_name = asset.ship_model.name_tag.lower().replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"episode_{episode_idx:03d}_{asset_name}.csv")
        if asset_idx == 0 and hasattr(asset, "logged_results_df"):
            asset.logged_results_df.to_csv(out_path, index=None)
        else:
            pd.DataFrame.from_dict(asset.ship_model.simulation_results).to_csv(out_path, index=None)


def plot_last_episode(assets, env, map_gdfs):
    repeat = False
    map_anim = MapAnimator(
        assets=assets,
        map_gdfs=map_gdfs,
        interval_ms=500,
        status_asset_index=0,
    )
    map_anim.run(fps=60, show=False, repeat=repeat)

    polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
    polar_anim.run(fps=60, show=False, repeat=repeat)

    animate_side_by_side(
        map_anim.fig,
        polar_anim.fig,
        left_frac=0.68,
        height_frac=0.92,
        gap_px=16,
        show=True,
    )

    result_dfs = []
    for idx, asset in enumerate(assets):
        if idx == 0 and hasattr(asset, "logged_results_df"):
            result_df = asset.logged_results_df.copy()
        else:
            result_df = pd.DataFrame.from_dict(asset.ship_model.simulation_results)
        result_dfs.append(result_df)
        plot_ship_status(asset, result_df, plot_env_load=(idx == 0), show=False)

    plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)
    plt.show()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    map_obj, map_gdfs = load_map_data()

    base_rng = np.random.default_rng(args.base_seed)
    campaign_rows = []
    last_assets = None
    last_env = None

    for episode_idx in range(1, args.n_episodes + 1):
        episode_seed = int(base_rng.integers(0, 2**31 - 1))
        row, assets, env, map_gdfs = run_episode(args, episode_idx, episode_seed, map_obj, map_gdfs)
        campaign_rows.append(row)
        last_assets = assets
        last_env = env

        if args.save_episode_results:
            save_episode_csvs(args, episode_idx, assets)

    summary_df = pd.DataFrame(campaign_rows)
    summary_csv = os.path.join(args.output_dir, "multi_ship_mc_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved campaign summary to: {summary_csv}")
    print(summary_df)

    if args.plot_results and last_assets is not None:
        plot_last_episode(last_assets, last_env, map_gdfs)


if __name__ == "__main__":
    main()
