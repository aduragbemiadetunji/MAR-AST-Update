from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from run_script.setup import get_env_assets

## IMPORT FUNCTIONS
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

## IMPORT AST RELATED TOOLS
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy, MultiInputPolicy
from gymnasium.wrappers import FlattenObservation, RescaleAction
from gymnasium.utils.env_checker import check_env

### IMPORT TOOLS
import argparse
import pandas as pd
import os
import time

### IMPORT UTILS
from utils.get_path import get_trained_model_path, get_saved_anim_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_cli_args():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Ship in Transit Simulation')

    ## Add arguments for environments
    parser.add_argument('--time_step', type=int, default=5, metavar='TIMESTEP',
                        help='ENV: time step size in second for ship transit simulator (default: 5)')
    parser.add_argument('--engine_step_count', type=int, default=10, metavar='ENGINE_STEP_COUNT',
                        help='ENV: engine integration step count in between simulation timestep (default: 10)')
    parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                        help='ENV: radius of acceptance in meter for LOS algorithm (default: 300)')
    parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                        help='ENV: lookahead distance in meter for LOS algorithm (default: 1000)')
    parser.add_argument('--nav_fail_time', type=int, default=600, metavar='NAV_FAIL_TIME',
                    help='ENV: Allowed recovery time in second from navigational failure warning condition (default: 600)')
    parser.add_argument('--traj_threshold_coeff', type=float, default=1.5, metavar='TRAJ_THRESHOLD_COEFF',
                    help='ENV: Coefficient to scale the maximum distance travelled based on the route segment length (default: 1.5)')
    parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                        help='ENV: record ship drawing for plotting and animation (default: True)')
    parser.add_argument('--time_since_last_ship_drawing', type=int, default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--map_gpkg_filename', type=str, default="Stangvik.gpkg", metavar='MAP_GPKG_FILENAME',
                        help='ENV: name of the .gpkg filename for the map (default: "Stangvik.gpkg")')
    parser.add_argument('--warm_up_time', type=int, default=2000, metavar='WARM_UP_TIME',
                        help='ENV: time needed in second before policy - action sampling takes place (default: 2000)')
    parser.add_argument('--action_sampling_period', type=int, default=1200, metavar='ACT_SAMPLING_PERIOD',
                        help='ENV: time period in second between policy - action sampling (default: 1200)')
    parser.add_argument('--max_sea_state', type=str, default="SS 5", metavar='MAX_SEA_STATE',
                        help='ENV: Maximum allowed sea state for environment model to condition the sea state table (default: "SS 5")')

    # Parse args
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

###################################### TRAIN THE MODEL #####################################
    # Path
    model_name  ="AST-train_2025-11-15_03-22-41_dd72"
    model_path, log_path = get_trained_model_path(root=ROOT, model_name=model_name)
    save_path = get_saved_anim_path(root=ROOT, model_name=model_name)
    
    # Get the args
    args = parse_cli_args()
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args, print_ship_status=True)
    
    # Set random route
    env.set_random_route_flag(flag=True)
    
    # Set for training flag
    env.set_for_training_flag(flag=False)
    
    # Load the trained model
    ast_model = SAC.load(model_path)
    
    ## Run the trained model
    obs, info = env.reset(seed=1, route_idx=0)
    while True:
        action, _states = ast_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
####################################### GET RESULTS ########################################

    # Print RL transition
    env.log_RL_transition_text(train_time=None,
                               txt_path=None,
                               also_print=True)

    ## Get the simulation results for all assets, and plot the asset simulation results
    own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
    result_dfs = [own_ship_results_df]
    print("len(sim_results) =", len(env.assets[0].ship_model.simulation_results["time [s]"]))


    # ############

    # # DO CBD HERE
    # from contracts.contracts import evaluate_contracts_over_dataframe, ViolationLogger
    # from contracts.logs.violation_summary_by_contract_table import summarize_violations_by_contract
    # logger = ViolationLogger("contracts/logs/contract_violations.csv", append=False)
    # evaluate_contracts_over_dataframe(own_ship_results_df, env, logger, run_id="baseline_run")
    # pivot_table = summarize_violations_by_contract("contracts/logs/contract_violations.csv")
    # print(pivot_table)
    # ###########

    # Build both animations (don’t show yet)
    repeat=False
    map_anim = MapAnimator(
        assets=assets,
        map_gdfs=map_gdfs,
        interval_ms=500,
        status_asset_index=0  # flags for own ship
    )
    map_anim.run(fps=120, show=False, repeat=False)

    polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
    polar_anim.run(fps=120, show=False, repeat=False)

    # Place windows next to each other, same height, centered
    animate_side_by_side(map_anim.fig, polar_anim.fig,
                        left_frac=0.68,  # how wide the map window is
                        height_frac=0.92,
                        gap_px=16,
                        show=True)

    # Plot 1: Status plot    
    plot_ship_status(env.assets[0], own_ship_results_df, plot_env_load=True, show=False)

    # Plot 2: Trajectory
    fig, ax = plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True, no_title=True)
    fig.savefig("ship_trajectory.pdf", bbox_inches="tight")
    # fig.savefig("ast_simulation_fail_validation_1.pdf", bbox_inches="tight")
    
    # Save animation
    save_anim = True
    save_anim = False
    if save_anim:
        id = 4
        map_file_name = f"{id}_map_anim.mp4"
        polar_file_name = f"{id}_polar_anim.mp4"
        map_anim.save(base_path=save_path, filename=map_file_name, fps=120)
        polar_anim.save(base_path=save_path, filename=polar_file_name, fps=120)