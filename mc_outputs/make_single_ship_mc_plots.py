import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def make_plots(csv_path, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("episode").reset_index(drop=True)

    for col in ["collision", "grounding", "nav_failure", "env_stop"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col].astype(str).str.strip().str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
                )
            df[col] = df[col].fillna(False).astype(bool)

    episodes = df["episode"].astype(int).tolist()
    x = np.arange(len(df))

    def savefig(name):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    # completion fraction
    if "completed_fraction" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(x, df["completed_fraction"])
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Completed fraction")
        plt.title("Mission completion fraction by episode")
        savefig("completion_fraction_by_episode.png")

    # supervisor burden
    if {"mode_switches", "dp_safe_entries"}.issubset(df.columns):
        plt.figure(figsize=(11, 5))
        width = 0.38
        plt.bar(x - width/2, df["mode_switches"], width=width, label="Mode switches")
        plt.bar(x + width/2, df["dp_safe_entries"], width=width, label="Drift entries")
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Count")
        plt.title("Fallback supervisory burden by episode")
        plt.legend()
        savefig("supervisory_burden_by_episode.png")

    # stacked contract violations
    stack_cols = [c for c in [
        "viol_HeadingController_A2",
        "viol_Machinery_A2",
        "viol_Machinery_G4",
        "viol_Rudder_G2",
        "viol_ThrottleController_G1",
    ] if c in df.columns]
    if stack_cols:
        plt.figure(figsize=(12, 6))
        bottom = np.zeros(len(df))
        labels_map = {
            "viol_HeadingController_A2": "HeadingController A2",
            "viol_Machinery_A2": "Machinery A2",
            "viol_Machinery_G4": "Machinery G4",
            "viol_Rudder_G2": "Rudder G2",
            "viol_ThrottleController_G1": "ThrottleController G1",
        }
        for col in stack_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
            plt.bar(x, vals, bottom=bottom, label=labels_map.get(col, col))
            bottom += vals
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Violation count")
        plt.title("Contract violations by episode and clause")
        plt.legend()
        savefig("contract_violations_stacked.png")

    # minimum engine availability
    if "min_available_engine_count" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(x, df["min_available_engine_count"])
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Minimum available engine count")
        plt.title("Minimum available engine count by episode")
        savefig("min_available_engine_count.png")

    # cross-track error
    if "max_abs_cross_track_error_m" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(x, df["max_abs_cross_track_error_m"])
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Maximum absolute cross-track error [m]")
        plt.title("Maximum cross-track error by episode")
        savefig("max_cross_track_error_by_episode.png")

    # fuel
    if "final_fuel_consumption_kg" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(x, df["final_fuel_consumption_kg"])
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Final fuel consumption [kg]")
        plt.title("Final fuel consumption by episode")
        savefig("final_fuel_by_episode.png")

    # scatter plots
    if {"mode_switches", "total_contract_violations"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["mode_switches"], df["total_contract_violations"])
        for _, row in df.iterrows():
            plt.annotate(str(int(row["episode"])), (row["mode_switches"], row["total_contract_violations"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Mode switches")
        plt.ylabel("Total contract violations")
        plt.title("Mode switches vs contract violations")
        savefig("scatter_mode_switches_vs_contract_violations.png")

    if {"mode_switches", "completed_fraction"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["mode_switches"], df["completed_fraction"])
        for _, row in df.iterrows():
            plt.annotate(str(int(row["episode"])), (row["mode_switches"], row["completed_fraction"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Mode switches")
        plt.ylabel("Completed fraction")
        plt.title("Mode switches vs mission completion fraction")
        savefig("scatter_mode_switches_vs_completion.png")

    if {"mode_switches", "final_fuel_consumption_kg"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["mode_switches"], df["final_fuel_consumption_kg"])
        for _, row in df.iterrows():
            plt.annotate(str(int(row["episode"])), (row["mode_switches"], row["final_fuel_consumption_kg"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Mode switches")
        plt.ylabel("Final fuel consumption [kg]")
        plt.title("Mode switches vs fuel consumption")
        savefig("scatter_mode_switches_vs_fuel.png")

    # outcome counts
    if any(c in df.columns for c in ["collision", "grounding", "nav_failure", "env_stop"]):
        status_counts = {
            "Collision": int(df["collision"].sum()) if "collision" in df.columns else 0,
            "Grounding": int(df["grounding"].sum()) if "grounding" in df.columns else 0,
            "Nav. failure": int(df["nav_failure"].sum()) if "nav_failure" in df.columns else 0,
            "Env. stop": int(df["env_stop"].sum()) if "env_stop" in df.columns else 0,
        }
        plt.figure(figsize=(7, 4.5))
        plt.bar(list(status_counts.keys()), list(status_counts.values()))
        plt.ylabel("Number of episodes")
        plt.title("Campaign-level safety outcome counts")
        savefig("outcome_counts.png")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("csv_path")
    # parser.add_argument("--out_dir", default=None)
    # args = parser.parse_args()
    csv_path = 'mc_outputs\\mc_summary.csv'
    out_dir = 'mc_outputs\\plots'
    make_plots(csv_path, out_dir)
