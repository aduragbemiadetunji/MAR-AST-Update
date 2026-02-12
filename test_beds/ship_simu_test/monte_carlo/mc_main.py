# mc_main.py
import numpy as np
import pandas as pd
from mc_scenarios import sample_scenario
from mc_runner import run_episode
from your_env_factory import make_env_and_assets  # you create this from your current script

def main():
    N = 200
    rng = np.random.default_rng(12345)

    rows = []

    # Create one env to read sim_time (or hardcode your sim time)
    env, assets, machinery_config, caps = make_env_and_assets()
    sim_time_s = assets[0].ship_model.int.sim_time

    for i in range(N):
        scenario = sample_scenario(rng, sim_time_s)
        # recreate env per episode if you later vary weather; for now you can reuse
        env, assets, machinery_config, caps = make_env_and_assets()

        row = run_episode(
            env=env,
            assets=assets,
            scenario=scenario,
            machinery_config=machinery_config,
            main_engine_capacity=caps["ME"],
            diesel_gen_capacity=caps["DG"],
            min_dwell_s=60.0,
        )
        row["episode"] = i + 1
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("mc_summary.csv", index=False)
    print(df.groupby("final_mode")["episode"].count())
    print(df.groupby("fault_mask")["stopped_early"].mean())

if __name__ == "__main__":
    main()
