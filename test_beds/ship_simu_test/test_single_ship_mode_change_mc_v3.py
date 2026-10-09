from pathlib import Path
import sys
import geopandas as gpd
import json

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from env_wrappers.multiship_env.env import AssetInfo, ShipAsset, MultiShipEnv

from simulator.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, SimulationConfiguration, ShipModel 
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
# from test_beds.ship_simu_test.monte_carlo.mc_faults import sample_fault_scenario
from test_beds.ship_simu_test.monte_carlo.mc_faults import (
    sample_fault_scenario,
    sample_multi_fault_scenario,
)

from contracts.contracts import evaluate_contracts_over_dataframe, ViolationLogger



### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


###############################################################################

# Argument Parser
parser = argparse.ArgumentParser(description='Ship in Transit Simulation')

## Add arguments for environments
parser.add_argument('--time_step', type=int, default=10, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 5)')
parser.add_argument('--engine_step_count', type=int, default=10, metavar='ENGINE_STEP_COUNT',
                    help='ENV: engine integration step count in between simulation timestep (default: 10)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance in meter for LOS algorithm (default: 300)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                    help='ENV: lookahead distance in meter for LOS algorithm (default: 1000)')
parser.add_argument('--nav_fail_time', type=int, default=1500, metavar='NAV_FAIL_TIME',
                    help='ENV: Allowed recovery time in second from navigational failure warning condition (default: 300)')
parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                    help='ENV: record ship drawing for plotting and animation (default: True)')
parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                    help='ENV: time delay in second between ship drawing record (default: 30)')

# Add arguments for AST-core
parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                    help='AST: number of simulation episode counts (default: 1)')

#Fault injection settings
parser.add_argument(
    '--fault_mode',
    type=str,
    default='none',
    choices=['none', 'deterministic', 'monte_carlo', 'fixed'], #fixed is for testing a specific fault scenario
    help='Fault injection mode'
)

parser.add_argument(
    '--n_faults',
    type=int,
    default=1,
    help='Number of randomly generated fault events for monte_carlo mode'
)

parser.add_argument(
    '--allow_fault_overlap',
    action='store_true',
    help='Allow randomly generated faults to overlap in time'
)


parser.add_argument(
    '--campaign_seed',
    type=int,
    default=10,
    help='Master seed used to generate per-episode seeds'
)

parser.add_argument(
    '--episode_seed',
    type=int,
    default=None,
    help='Replay a specific episode seed exactly; mainly for n_episodes=1'
)

parser.add_argument(
    '--save_dir',
    type=str,
    default='mc_outputs',
    help='Directory for Monte Carlo summaries and per-episode logs'
)

parser.add_argument(
    '--save_timeseries',
    action='store_true',
    help='Save full timestep CSV for every episode'
)

parser.add_argument(
    '--timeseries_every',
    type=int,
    default=0,
    help='Save every Nth episode timestep CSV; 0 disables periodic saving'
)

parser.add_argument(
    '--plot_final_run',
    action='store_true',
    help='Plot the final/replayed run'
)

parser.add_argument(
    '--enable_fallback',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Enable fallback behavior when primary control mode fails'
)
parser.add_argument("--simulation_time", type=float, default=35000.0, metavar="SIM_TIME",
                        help="Simulation time in seconds for all assets")


#EDITING: parse more arguments
parser.add_argument("--scenario_start_s", type=float, default=2000.0)
parser.add_argument("--fault_delay_s", type=float, default=500.0)
parser.add_argument("--fixed_fault_duration_s", type=float, default=1000.0)
parser.add_argument("--fixed_fault_mask", type=str, default="ME")
parser.add_argument("--post_fault_observation_s", type=float, default=900.0)
parser.add_argument("--terminate_after_fault_response", action="store_true")




args = parser.parse_args()

# -----------------------
# GPKG settings (edit if your layer names differ)
# -----------------------
GPKG_PATH   = get_map_path(ROOT, "Trondheim_Kristiansund.gpkg")       # <-- put your file here (or absolute path)
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"               # optional
WATER_LAYER = None#"water_3857"               # optional
FERRY_ROUTE_LAYER  = None #"ferry_routes_3857" #OPTIONAL


frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER, FERRY_ROUTE_LAYER)
map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf

map_data = get_polygon_from_gdf(land_gdf)   # list of exterior rings (E,N)
map = [PolygonObstacle(map_data), frame_gdf]              # <-- reuse your existing simulator map type

# Engine configuration
main_engine_capacity = 2160e3 #4160e3
diesel_gen_capacity = 510e3 #610e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

# Configure the simulation
ship_config = ShipConfiguration(
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
    nonlinear_friction_coefficient__in_yaw=400
)
wave_model_config = WaveModelConfiguration(
    minimum_wave_frequency=0.4,
    maximum_wave_frequency=2.5,
    wave_frequency_discrete_unit_count=50,
    minimum_spreading_angle=-np.pi,
    maximum_spreading_angle=np.pi,
    spreading_angle_discrete_unit_count=10,
    spreading_coefficient=1,
    rho=1025.0,
    timestep_size=args.time_step
)
current_model_config = CurrentModelConfiguration(
    initial_current_velocity=0.01,
    current_velocity_standard_deviation=0.0075,
    current_velocity_decay_rate=0.025,
    initial_current_direction=np.deg2rad(0.0),
    current_direction_standard_deviation=0.025,
    current_direction_decay_rate=0.025,
    timestep_size=args.time_step
)
wind_model_config = WindModelConfiguration(
    initial_mean_wind_velocity=None,                    # Set to None to use a mean wind component
    mean_wind_velocity_decay_rate=0.025,
    mean_wind_velocity_standard_deviation=0.005,
    initial_wind_direction=np.deg2rad(0.0),
    wind_direction_decay_rate=0.025,
    wind_direction_standard_deviation=0.025,
    minimum_mean_wind_velocity=0.0,
    maximum_mean_wind_velocity=42.0,
    minimum_wind_gust_frequency=0.06,
    maximum_wind_gust_frequency=0.4,
    wind_gust_frequency_discrete_unit_count=100,
    clip_speed_nonnegative=True,
    kappa_parameter=0.0026,
    U10=10.0,
    wind_evaluation_height=5.0,
    timestep_size=args.time_step
)
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator,
    name_tag='PTO'
)
pto_mode = MachineryMode(params=pto_mode_params)

pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2*diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor,
    name_tag='PTI'
)
pti_mode = MachineryMode(params=pti_mode_params)

mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline,
    name_tag='MEC'
)
mec_mode = MachineryMode(params=mec_mode_params)

# I can write more modes here that kindof indicate what the fall back strategies would be
######MORE MODES########

pti_1dg_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=1 * diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor,
    name_tag='PTI_1DG'
)
pti_1dg_mode = MachineryMode(params=pti_1dg_params)


mec_blackout_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_offline,
    name_tag='MEC_BLACKOUT'
)
mec_blackout_mode = MachineryMode(params=mec_blackout_params)

dp_safe_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_offline,
    name_tag='DP_SAFE'
)
dp_safe_mode = MachineryMode(params=dp_safe_params)




mso_modes = MachineryModes([
    pto_mode, 
    mec_mode, 
    pti_mode,
    pti_1dg_mode,
    mec_blackout_mode,
    dp_safe_mode
    ])

fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=0,
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200e3,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=35,
    max_rudder_rate_degree_per_s=2.3,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

### CONFIGURE THE SHIP SIMULATION MODELS
## Own ship
own_ship_route_filename = 'Trondheim_Kristiansund_OS.txt'
own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)

start_E, start_N = np.loadtxt(own_ship_route_name)[0]  # expecting two columns: east, north

own_ship_config = SimulationConfiguration(
    initial_north_position_m=start_E,
    initial_east_position_m=start_N,
    initial_yaw_angle_rad=np.deg2rad(0.0),
    initial_forward_speed_m_per_s=0.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=args.simulation_time,
)
# Set the throttle and autopilot controllers for the own ship
own_ship_throttle_controller_gains = ThrottleControllerGains(
   kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001
)

own_ship_heading_controller_gains = HeadingControllerGains(kp=1.5, kd=75, ki=0.005)
own_ship_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
own_ship_desired_speed = 4.0
own_ship_cross_track_error_tolerance = 3000
own_ship_initial_propeller_shaft_speed = 0
own_ship_initial_propeller_shaft_acceleration = 0
own_ship = ShipModel(
    ship_config=ship_config,
    simulation_config=own_ship_config,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,                       
    throttle_controller_gain=own_ship_throttle_controller_gains, #None
    heading_controller_gain=own_ship_heading_controller_gains, #None
    los_parameters=own_ship_los_guidance_parameters, #None
    name_tag='Own ship',
    route_name=own_ship_route_name,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=own_ship_initial_propeller_shaft_speed * np.pi /30,
    initial_propeller_shaft_acc_rad_per_sec2=own_ship_initial_propeller_shaft_acceleration * np.pi / 30,
    desired_speed=own_ship_desired_speed,
    cross_track_error_tolerance=own_ship_cross_track_error_tolerance,
    nav_fail_time=args.nav_fail_time,
    map_obj=map[0],
    colav_mode=None,
    print_status=True
)



own_ship_info = AssetInfo(
    # dynamic state (mutable)
    current_north       = own_ship.north,
    current_east        = own_ship.east,
    current_yaw_angle   = own_ship.yaw_angle,
    forward_speed       = own_ship.forward_speed,
    sideways_speed      = own_ship.sideways_speed,

    # static properties (constants)
    name_tag            = own_ship.name_tag,
    ship_length         = own_ship.l_ship,
    ship_width          = own_ship.w_ship
)
# Wraps simulation objects based on the ship type using a dictionary
own_ship_asset = ShipAsset(
    ship_model=own_ship,
    info=own_ship_info
)

own_ship_asset.route_name = own_ship_route_name
own_ship_asset.route_points = np.atleast_2d(np.loadtxt(own_ship_route_name))[:, :2]

# Package the assets for reinforcement learning agent
assets: List[ShipAsset] = [own_ship_asset]

################################### ENV SPACE ###################################

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipEnv(
    assets=assets,
    map=map,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    args=args,
    include_wave=True,
    include_wind=True,
    include_current=True)



def safe_last(df, col, default=np.nan):
    if col in df.columns and len(df[col]) > 0:
        return df[col].iloc[-1]
    return default


def safe_max_abs(df, col, default=np.nan):
    if col in df.columns and len(df[col]) > 0:
        return float(np.nanmax(np.abs(df[col].to_numpy())))
    return default


def safe_min(df, col, default=np.nan):
    if col in df.columns and len(df[col]) > 0:
        return float(np.nanmin(df[col].to_numpy()))
    return default


def safe_max(df, col, default=np.nan):
    if col in df.columns and len(df[col]) > 0:
        return float(np.nanmax(df[col].to_numpy()))
    return default


def scenario_to_jsonable(scenario):
    if scenario is None:
        return {"type": "none"}

    if hasattr(scenario, "events"):
        return {
            "type": "multi_fault",
            "events": [
                {
                    "start_s": float(ev.start_s),
                    "end_s": float(ev.end_s),
                    "duration_s": float(ev.duration_s),
                    "fault_mask": dict(ev.fault_mask),
                }
                for ev in scenario.events
            ]
        }

    return {
        "type": "single_fault",
        "start_s": float(getattr(scenario, "start_s", np.nan)),
        "duration_s": float(getattr(scenario, "duration_s", 0.0)),
        "fault_mask": dict(getattr(scenario, "fault_mask", {})),
    }


def summarize_violations(viol_df):
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


class NoFaultScenario:
    def __init__(self):
        self.start_s = np.nan
        self.duration_s = 0.0
        self.fault_mask = (False, False, False, False)

    def engine_status(self, t):
        return {"ME": True, "DG1": True, "DG2": True, "HSG": True}

import inspect

### THIS IS WHERE THE EPISODE HAPPENS

# episode = 1
# base_rng = np.random.default_rng(10)

episode = 1
base_rng = np.random.default_rng(args.campaign_seed)

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

episode_summary_rows = []

last_episode_df = None
last_assets = None
last_result_dfs = None



class FixedFaultScenario:
    def __init__(self, start_s, duration_s, failed_components):
        self.start_s = float(start_s)
        self.duration_s = float(duration_s)
        self.end_s = self.start_s + self.duration_s
        self.fault_mask = {
            "ME": "ME" in failed_components,
            "DG1": "DG1" in failed_components,
            "DG2": "DG2" in failed_components,
            "HSG": "HSG" in failed_components,
        }

    def engine_status(self, t):
        status = {"ME": True, "DG1": True, "DG2": True, "HSG": True}

        if self.start_s <= t < self.end_s:
            for component, failed in self.fault_mask.items():
                if failed:
                    status[component] = False

        return status


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


while episode <= args.n_episodes:
    # Reset the environment at the beginning of episode
    env.reset()


    if args.episode_seed is not None:
        # replay mode: use the provided episode seed exactly
        episode_seed = int(args.episode_seed)
    else:
        # campaign mode: derive one seed per episode from campaign_seed
        episode_seed = int(base_rng.integers(0, 2**31 - 1))

    episode_rng = np.random.default_rng(episode_seed)
    
    ## THIS IS WHERE THE SIMULATION HAPPENS
    mach = assets[0].ship_model.ship_machinery_model



    # Dwell-time tracking
    last_mode = mach.operating_mode
    last_mode_change_time = 0.0  # seconds
    MIN_DWELL_S = 60.0           # tune as needed


    # -----------------------------
    # History logs for plotting
    # -----------------------------
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

    nominal_desired_speed = float(own_ship_desired_speed)
    desired_speed_history = []
    propulsion_fraction_history = []

    running_time = 0

    
    # scenario = sample_fault_scenario(base_rng, sim_time_s=own_ship.int.sim_time)
    # print(f"[EP {episode}] scenario: start={scenario.start_s:.1f}s dur={scenario.duration_s:.1f}s mask={scenario.fault_mask}")

    # if args.fault_mode == "none":
    #     scenario = NoFaultScenario()
    #     print(f"[EP {episode}] no-fault baseline run")

    # elif args.fault_mode == "deterministic":
    #     scenario = sample_fault_scenario(base_rng, sim_time_s=own_ship.int.sim_time)
    #     print(
    #         f"[EP {episode}] deterministic fault: "
    #         f"start={scenario.start_s:.1f}s dur={scenario.duration_s:.1f}s mask={scenario.fault_mask}"
    #     )

    # elif args.fault_mode == "monte_carlo":
    #     scenario = sample_multi_fault_scenario(
    #         base_rng,
    #         sim_time_s=own_ship.int.sim_time,
    #         n_faults=args.n_faults,
    #         # min_gap_s=120.0,
    #         # max_gap_s=500.0,
    #         # duration_range_s=(120.0, 450.0),
    #         # allow_overlap=args.allow_fault_overlap,

    #         start_frac=(0.08, 0.92),
    #         duration_range_s=(150.0, 500.0),
    #         # min_separation_s=2500.0,
    #     )

    #     print(f"[EP {episode}] MC multi-fault scenario with {len(scenario.events)} events")
    #     for i, ev in enumerate(scenario.events, start=1):
    #         print(
    #             f"    event {i}: start={ev.start_s:.1f}s "
    #             f"end={ev.end_s:.1f}s dur={ev.duration_s:.1f}s "
    #             f"mask={ev.fault_mask}"
    #         )

        
    if args.fault_mode == "none":
        scenario = NoFaultScenario()
        print(f"[EP {episode}] no-fault baseline run | episode_seed={episode_seed}")

    elif args.fault_mode == "deterministic":
        scenario = sample_fault_scenario(episode_rng, sim_time_s=own_ship.int.sim_time)
        print(
            f"[EP {episode}] deterministic fault | episode_seed={episode_seed} | "
            f"start={scenario.start_s:.1f}s dur={scenario.duration_s:.1f}s mask={scenario.fault_mask}"
        )

    elif args.fault_mode == "monte_carlo":
        scenario = sample_multi_fault_scenario(
            episode_rng,
            sim_time_s=own_ship.int.sim_time,
            n_faults=args.n_faults,
            start_frac=(0.08, 0.92),
            duration_range_s=(150.0, 500.0),
        )

        print(
            f"[EP {episode}] MC multi-fault scenario | "
            f"episode_seed={episode_seed} | n_events={len(scenario.events)}"
        )

        for i, ev in enumerate(scenario.events, start=1):
            print(
                f"    event {i}: start={ev.start_s:.1f}s "
                f"end={ev.end_s:.1f}s dur={ev.duration_s:.1f}s "
                f"mask={ev.fault_mask}"
            )

    elif args.fault_mode == "fixed":
        failed_components = {
            item.strip().upper()
            for item in args.fixed_fault_mask.split(",")
            if item.strip()
        }

        fault_start_abs_s = args.scenario_start_s + args.fault_delay_s

        scenario = FixedFaultScenario(
            start_s=fault_start_abs_s,
            duration_s=args.fixed_fault_duration_s,
            failed_components=failed_components,
        )

        print(
            f"[EP {episode}] fixed local disturbance | "
            f"scenario_start={args.scenario_start_s:.1f}s | "
            f"fault_start_abs={scenario.start_s:.1f}s | "
            f"fault_delay={args.fault_delay_s:.1f}s | "
            f"duration={scenario.duration_s:.1f}s | "
            f"mask={scenario.fault_mask}"
        )



    # Initial engine availability (later: this will be updated by fault injection)
    engine_status = {"ME": True, "DG1": True, "DG2": True, "HSG": True}
    while running_time < own_ship.int.sim_time and env.stop is False:
        engine_status = scenario.engine_status(running_time) #update engine status based on the scenario and current time, toggle to enable fault injection
        prop_frac = propulsion_fraction_from_mode(mach.operating_mode, engine_status)
        # Fault status for logging
        # fault_active = False
        # if hasattr(scenario, "start_s") and hasattr(scenario, "duration_s"):
        #     fault_active = (scenario.start_s <= running_time < scenario.start_s + scenario.duration_s)

        # fault_start_s = getattr(scenario, "start_s", np.nan)
        # fault_end_s = getattr(scenario, "start_s", np.nan) + getattr(scenario, "duration_s", 0.0)

        fault_active = False
        fault_start_s = np.nan
        fault_end_s = np.nan
        active_fault_count = 0

        # if hasattr(scenario, "active_events"):
        #     active_events = scenario.active_events(running_time)
        #     active_fault_count = len(active_events)
        #     fault_active = active_fault_count > 0

        #     if fault_active:
        #         fault_start_s = min(ev.start_s for ev in active_events)
        #         fault_end_s = max(ev.end_s for ev in active_events)

        # else:
        #     if np.isfinite(getattr(scenario, "start_s", np.nan)):
        #         fault_start_s = scenario.start_s
        #         fault_end_s = scenario.start_s + getattr(scenario, "duration_s", 0.0)
        #         fault_active = (fault_start_s <= running_time < fault_end_s)
        #         active_fault_count = int(fault_active)

        if hasattr(scenario, "active_events"):
            active_events = scenario.active_events(running_time)
            fault_active = len(active_events) > 0
            active_fault_count = len(active_events)

            if fault_active:
                fault_start_s = min(ev.start_s for ev in active_events)
                fault_end_s = max(ev.end_s for ev in active_events)
            else:
                fault_start_s = np.nan
                fault_end_s = np.nan

        else:
            fault_active = False
            active_fault_count = 0

            if hasattr(scenario, "start_s") and hasattr(scenario, "duration_s"):
                fault_active = (scenario.start_s <= running_time < scenario.start_s + scenario.duration_s)
                active_fault_count = int(fault_active)
                fault_start_s = getattr(scenario, "start_s", np.nan)
                fault_end_s = getattr(scenario, "start_s", np.nan) + getattr(scenario, "duration_s", 0.0)
            else:
                fault_start_s = np.nan
                fault_end_s = np.nan



        # env.step(turn_off_engine=turn_off_engine)
        env.step()

        # Update running time
        # running_time = np.max([asset.ship_model.int.time for asset in assets])

        # --- build "power demand" (simple placeholder, refine later) ---
        required_hotel_power = machinery_config.hotel_load
        # required_propulsion_power = 0.6 * main_engine_capacity if assets[0].ship_model.desired_speed > 0 else 0.0
        required_propulsion_power = 1000 * assets[0].ship_model.simulation_results['propulsion power [kw]'][-1]
        # print(assets[0].ship_model.simulation_results['propulsion power [kw]'][-1])


        # --- dwell time ---
        time_in_mode = running_time - last_mode_change_time

        # --- decision ---
        decision = resilience_supervisor(
            current_mode=mach.operating_mode,
            engine_status=engine_status,
            required_propulsion_power=required_propulsion_power,
            required_hotel_power=required_hotel_power,
            me_capacity=main_engine_capacity,
            dg_capacity=diesel_gen_capacity,
            min_dwell_s=MIN_DWELL_S,
            time_in_mode_s=time_in_mode,
        )

        if not args.enable_fallback:
            # No resilience manager: keep the same mode, but degrade propulsion
            own_ship.desired_speed = nominal_desired_speed * prop_frac

        else:
            own_ship.desired_speed = nominal_desired_speed
            if decision.action in ("SWITCH_MODE", "ENTER_SAFE_STATE"):
                if mach.operating_mode != decision.recommended_mode:
                    print(f"[t={running_time:.1f}s] {decision.action}: {mach.operating_mode} -> {decision.recommended_mode} | {decision.reason}")
                    mach.operating_mode = decision.recommended_mode
                    last_mode_change_time = running_time
            apply_safe_state_overrides(assets[0].ship_model)


        # -----------------------------
        # Append histories for plotting
        # -----------------------------
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

        active_fault_count_history.append(active_fault_count)

        significant_wave_height.append(env.Hs)
        wave_period.append(env.Tp)

        desired_speed_history.append(float(own_ship.desired_speed))
        propulsion_fraction_history.append(float(prop_frac))


        # Update running time after stepping
        running_time = np.max([asset.ship_model.int.time for asset in assets])

        fault_start_abs_s = args.scenario_start_s + args.fault_delay_s
        fault_end_abs_s = fault_start_abs_s + args.fixed_fault_duration_s
        episode_stop_s = fault_end_abs_s + args.post_fault_observation_s


        if args.terminate_after_fault_response and running_time >= episode_stop_s:
            # max_response_end_s = scenario.end_s + args.post_fault_observation_s
            # if running_time >= max_response_end_s:
            print(
                f"[EP {episode}] terminating local disturbance scenario at "
                f"t={running_time:.1f}s "
                f"(fault_end={scenario.end_s:.1f}s, "
                f"post_fault_observation={args.post_fault_observation_s:.1f}s)"
            )
            break


    own_ship_results_df = pd.DataFrame.from_dict(env.assets[0].ship_model.simulation_results)

    n = len(own_ship_results_df)

    own_ship_results_df["operating mode"] = mode_history[:n]
    own_ship_results_df["fault active"] = fault_active_history[:n]
    own_ship_results_df["fault start [s]"] = fault_start_history[:n]
    own_ship_results_df["fault end [s]"] = fault_end_history[:n]

    own_ship_results_df["ME available"] = me_status_history[:n]
    own_ship_results_df["DG1 available"] = dg1_status_history[:n]
    own_ship_results_df["DG2 available"] = dg2_status_history[:n]
    own_ship_results_df["HSG available"] = hsg_status_history[:n]

    own_ship_results_df["decision action"] = decision_action_history[:n]
    own_ship_results_df["decision reason"] = decision_reason_history[:n]
    own_ship_results_df["active fault count"] = active_fault_count_history[:n]

    own_ship_results_df["available engine count"] = (
        own_ship_results_df["ME available"].astype(int)
        + own_ship_results_df["DG1 available"].astype(int)
        + own_ship_results_df["DG2 available"].astype(int)
        + own_ship_results_df["HSG available"].astype(int)
    )

    own_ship_results_df["episode"] = episode
    own_ship_results_df["episode_seed"] = episode_seed

    own_ship_results_df["significant wave height [m]"] = significant_wave_height[:n]
    own_ship_results_df["wave period [s]"] = wave_period[:n]


    own_ship_results_df["desired speed [m/s]"] = desired_speed_history[:n]
    own_ship_results_df["propulsion fraction [-]"] = propulsion_fraction_history[:n]

    # Add contract-monitor aliases so the updated runner can use logged
    # machinery availability and mode history without changing existing plots.
    own_ship_results_df["operating mode [str]"] = own_ship_results_df["operating mode"].astype(str)
    own_ship_results_df["ME healthy [-]"] = own_ship_results_df["ME available"].astype(int)
    own_ship_results_df["DG1 healthy [-]"] = own_ship_results_df["DG1 available"].astype(int)
    own_ship_results_df["DG2 healthy [-]"] = own_ship_results_df["DG2 available"].astype(int)
    own_ship_results_df["HSG healthy [-]"] = own_ship_results_df["HSG available"].astype(int)

    viol_csv_path = save_dir / f"violations_episode_{episode:03d}_seed_{episode_seed}.csv"
    logger = ViolationLogger(str(viol_csv_path), append=False)


    scenario_df = own_ship_results_df[
        (own_ship_results_df["time [s]"] >= args.scenario_start_s)
        & (own_ship_results_df["time [s]"] <= episode_stop_s)
    ].copy()

    scenario_df["scenario time [s]"] = (
        scenario_df["time [s]"] - args.scenario_start_s
    )


    evaluate_contracts_over_dataframe(
        # own_ship_results_df,#for entire dataframe
        scenario_df, #for scenario dataframe
        env,
        logger,
        run_id=f"episode_{episode:03d}_seed_{episode_seed}"
    )

    try:
        viol_df = pd.read_csv(viol_csv_path)
    except Exception:
        viol_df = pd.DataFrame()


    print(len(own_ship_results_df))
    print('----------------------')
    print(len(scenario_df))

    

    # scenario_json_path = save_dir / f"scenario_episode_{episode:03d}_seed_{episode_seed}.json"
    # with open(scenario_json_path, "w", encoding="utf-8") as f:
    #     json.dump(scenario_to_jsonable(scenario), f, indent=2)


    mode_series = own_ship_results_df["operating mode"].astype(str)
    mode_switches = int((mode_series != mode_series.shift(1)).sum() - 1)
    dp_safe_entries = int(((mode_series == "DP_SAFE") & (mode_series.shift(1) != "DP_SAFE")).sum())

    summary_row = {
        "episode": episode,
        "campaign_seed": args.campaign_seed,
        "episode_seed": episode_seed,
        "fault_mode": args.fault_mode,
        "n_faults_requested": args.n_faults if args.fault_mode == "monte_carlo" else int(args.fault_mode == "deterministic"),
        "n_faults_realized": len(scenario.events) if hasattr(scenario, "events") else int(args.fault_mode == "deterministic"),
        "sim_time_limit_s": float(own_ship.int.sim_time),
        "final_time_s": safe_last(own_ship_results_df, "time [s]"),
        "completed_fraction": (
            safe_last(own_ship_results_df, "time [s]") / float(own_ship.int.sim_time)
            if "time [s]" in own_ship_results_df.columns else np.nan
        ),
        "collision": bool(getattr(env, "collision", False)),
        "grounding": bool(getattr(env, "grounding", False)),
        "nav_failure": bool(getattr(env, "navigation_failure", False)),
        "env_stop": bool(env.stop),
        "mode_switches": mode_switches,
        "dp_safe_entries": dp_safe_entries,
        "max_available_engine_count": safe_max(own_ship_results_df, "available engine count"),
        "min_available_engine_count": safe_min(own_ship_results_df, "available engine count"),
        "max_active_fault_count": safe_max(own_ship_results_df, "active fault count"),
        "max_abs_cross_track_error_m": safe_max_abs(own_ship_results_df, "cross track error [m]"),
        "max_forward_speed_mps": safe_max(own_ship_results_df, "forward speed [m/s]"),
        "min_forward_speed_mps": safe_min(own_ship_results_df, "forward speed [m/s]"),
        "final_fuel_consumption_kg": safe_last(own_ship_results_df, "fuel consumption [kg]"),
    }

    summary_row.update(summarize_violations(viol_df))
    episode_summary_rows.append(summary_row)


    should_save_timeseries = False
    if args.save_timeseries:
        should_save_timeseries = True
    elif args.timeseries_every > 0 and (episode % args.timeseries_every == 0):
        should_save_timeseries = True
    elif args.episode_seed is not None:
        should_save_timeseries = True

    if should_save_timeseries:
        timeseries_path = save_dir / f"timeseries_episode_{episode:03d}_seed_{episode_seed}.csv"
        own_ship_results_df.to_csv(timeseries_path, index=False)


    # scenario_timeseries_path = save_dir / f"scenario_timeseries_episode_{episode:03d}_seed_{episode_seed}.csv"
    # scenario_df.to_csv(scenario_timeseries_path, index=False)
    


    last_episode_df = own_ship_results_df.copy()
    last_assets = assets
    last_result_dfs = [own_ship_results_df]

    # Increment the episode
    episode += 1



mc_summary_df = pd.DataFrame(episode_summary_rows)
mc_summary_path = save_dir / "mc_summary.csv"
mc_summary_df.to_csv(mc_summary_path, index=False)

print("\nSaved Monte Carlo summary to:", mc_summary_path)
print(mc_summary_df.head())


if args.plot_final_run or args.episode_seed is not None or args.n_episodes == 1:
    if last_episode_df is not None:
        repeat = False

        map_anim = MapAnimator(
            assets=assets,
            map_gdfs=map_gdfs,
            interval_ms=500,
            status_asset_index=0
        )
        map_anim.run(fps=120, show=False, repeat=False)

        polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
        polar_anim.run(fps=240, show=False, repeat=False)

        animate_side_by_side(
            map_anim.fig,
            polar_anim.fig,
            left_frac=0.68,
            height_frac=0.92,
            gap_px=16,
            show=True
        )

        plot_ship_status(own_ship_asset, last_episode_df, plot_env_load=False, show=True)
        plot_ship_and_real_map(assets, last_result_dfs, map_gdfs, show=True, profile="full")
else:
    print("Skipping plots. Use --episode_seed <seed> --plot_final_run to replay interesting scenarios.")








# ################################## GET RESULTS ##################################

# ## Get the simulation results for all assets, and plot the asset simulation results
# own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
# # Attach history logs to dataframe
# n = len(own_ship_results_df)

# own_ship_results_df["operating mode"] = mode_history[:n]
# own_ship_results_df["fault active"] = fault_active_history[:n]
# own_ship_results_df["fault start [s]"] = fault_start_history[:n]
# own_ship_results_df["fault end [s]"] = fault_end_history[:n]

# own_ship_results_df["ME available"] = me_status_history[:n]
# own_ship_results_df["DG1 available"] = dg1_status_history[:n]
# own_ship_results_df["DG2 available"] = dg2_status_history[:n]
# own_ship_results_df["HSG available"] = hsg_status_history[:n]

# own_ship_results_df["decision action"] = decision_action_history[:n]
# own_ship_results_df["decision reason"] = decision_reason_history[:n]

# own_ship_results_df["active fault count"] = active_fault_count_history[:n]

# mode_map = {
#     "PTO": 0,
#     "MEC": 1,
#     "PTI": 2,
#     "PTI_1DG": 3,
#     "MEC_BLACKOUT": 4,
#     "DP_SAFE": 5,
# }
# own_ship_results_df["operating mode code"] = own_ship_results_df["operating mode"].map(mode_map)

# own_ship_results_df["available engine count"] = (
#     own_ship_results_df["ME available"] +
#     own_ship_results_df["DG1 available"] +
#     own_ship_results_df["DG2 available"] +
#     own_ship_results_df["HSG available"]
# )

# result_dfs = [own_ship_results_df]


# ############

# # DO CBD HERE
# from contracts.contracts import evaluate_contracts_over_dataframe, ViolationLogger
# from contracts.logs.violation_summary_by_contract_table import summarize_violations_by_contract
# logger = ViolationLogger("contracts/logs/contract_violations.csv", append=False)
# evaluate_contracts_over_dataframe(own_ship_results_df, env, logger, run_id="baseline_run")
# # pivot_table = summarize_violations_by_contract("contracts/logs/contract_violations.csv")
# # print(pivot_table)

# # ###########


# # Build both animations (don’t show yet)
# repeat=False
# map_anim = MapAnimator(
#     assets=assets,
#     map_gdfs=map_gdfs,
#     interval_ms=500,
#     status_asset_index=0  # flags for own ship
# )
# map_anim.run(fps=120, show=False, repeat=False)

# polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
# polar_anim.run(fps=240, show=False, repeat=False)

# # Place windows next to each other, same height, centered
# animate_side_by_side(map_anim.fig, polar_anim.fig,
#                      left_frac=0.68,  # how wide the map window is
#                      height_frac=0.92,
#                      gap_px=16,
#                      show=True)

# # Plot 1: Trajectory
# plot_ship_status(own_ship_asset, own_ship_results_df, plot_env_load=True, show=True)

# # Plot 2: Status plot
# plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)