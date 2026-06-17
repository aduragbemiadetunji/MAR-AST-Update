import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

def parse_fault_windows(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []

def unavailable_count(mask):
    if not isinstance(mask, dict):
        return np.nan
    return sum(1 for v in mask.values() if v is False)

def make_multi_ship_plots(csv_path, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("episode").reset_index(drop=True)

    if "env_stop" in df.columns:
        if df["env_stop"].dtype == object:
            df["env_stop"] = (
                df["env_stop"].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
            )
        df["env_stop"] = df["env_stop"].fillna(False).astype(bool)

    fault_total_duration = []
    fault_max_unavailable = []
    for item in df["fault_windows"]:
        events = parse_fault_windows(item)
        durations = [float(ev.get("duration_s", 0.0)) for ev in events]
        unavailable = [unavailable_count(ev.get("fault_mask", {})) for ev in events]
        fault_total_duration.append(float(np.sum(durations)) if durations else 0.0)
        fault_max_unavailable.append(int(np.max(unavailable)) if unavailable else 0)

    df["fault_total_duration_s"] = fault_total_duration
    df["fault_max_unavailable_engines"] = fault_max_unavailable

    episodes = df["episode"].astype(int).tolist()
    x = np.arange(len(df))

    def savefig(name):
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=200, bbox_inches="tight")
        plt.close()

    def save_bar(series, ylabel, title, filename):
        plt.figure(figsize=(10, 5))
        plt.bar(x, series)
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        savefig(filename)

    def save_scatter(xdata, ydata, xlabel, ylabel, title, filename):
        plt.figure(figsize=(8, 5))
        plt.scatter(xdata, ydata)
        for _, row in df.iterrows():
            plt.annotate(str(int(row["episode"])), (xdata.loc[row.name], ydata.loc[row.name]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        savefig(filename)

    if "sim_completion_fraction" in df.columns:
        save_bar(df["sim_completion_fraction"], "Completion fraction",
                 "Mission completion fraction by episode", "completion_fraction_by_episode.png")

    if "mode_switch_count" in df.columns:
        save_bar(df["mode_switch_count"], "Mode-switch count",
                 "Fallback supervisory burden by episode", "mode_switches_by_episode.png")

    if "own_ship_max_cross_track_error_m" in df.columns:
        save_bar(df["own_ship_max_cross_track_error_m"], "Maximum cross-track error [m]",
                 "Own-ship maximum cross-track error by episode", "max_cross_track_error_by_episode.png")

    if "own_ship_fuel_consumption_kg" in df.columns:
        save_bar(df["own_ship_fuel_consumption_kg"], "Fuel consumption [kg]",
                 "Own-ship final fuel consumption by episode", "final_fuel_by_episode.png")

    if "min_available_engine_count" in df.columns:
        save_bar(df["min_available_engine_count"], "Minimum available engine count",
                 "Minimum available engine count by episode", "min_available_engine_count.png")

    save_bar(df["fault_total_duration_s"], "Total fault duration [s]",
             "Total faulted duration by episode", "total_fault_duration_by_episode.png")

    stack_cols = [c for c in [
        "viol_HeadingController_A2",
        "viol_Machinery_A2",
        "viol_Machinery_G4",
        "viol_Rudder_G2",
        "viol_SystemLevel_G5",
        "viol_ThrottleController_G1",
    ] if c in df.columns]
    if stack_cols:
        plt.figure(figsize=(12, 6))
        bottom = np.zeros(len(df))
        for col in stack_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
            plt.bar(x, vals, bottom=bottom, label=col.replace("viol_", "").replace("_", " "))
            bottom += vals
        plt.xticks(x, episodes)
        plt.xlabel("Episode")
        plt.ylabel("Violation count")
        plt.title("Contract violations by episode and clause")
        plt.legend()
        savefig("contract_violations_stacked.png")

    if {"mode_switch_count", "total_contract_violations"}.issubset(df.columns):
        save_scatter(df["mode_switch_count"], df["total_contract_violations"],
                     "Mode-switch count", "Total contract violations",
                     "Mode switches vs contract violations",
                     "scatter_mode_switches_vs_contract_violations.png")

    if {"fault_total_duration_s", "total_contract_violations"}.issubset(df.columns):
        save_scatter(df["fault_total_duration_s"], df["total_contract_violations"],
                     "Total fault duration [s]", "Total contract violations",
                     "Fault duration vs contract violations",
                     "scatter_fault_duration_vs_contract_violations.png")

    if {"mode_switch_count", "sim_completion_fraction"}.issubset(df.columns):
        save_scatter(df["mode_switch_count"], df["sim_completion_fraction"],
                     "Mode-switch count", "Completion fraction",
                     "Mode switches vs mission completion fraction",
                     "scatter_mode_switches_vs_completion.png")

    if {"min_available_engine_count", "own_ship_max_cross_track_error_m"}.issubset(df.columns):
        save_scatter(df["min_available_engine_count"], df["own_ship_max_cross_track_error_m"],
                     "Minimum available engine count", "Maximum cross-track error [m]",
                     "Minimum available engine count vs cross-track error",
                     "scatter_min_engines_vs_cte.png")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("csv_path")
    # parser.add_argument("--out_dir", default=None)
    # args = parser.parse_args()
    csv_path = "multi_ship_mc_outputs/multi_ship_mc_summary.csv"
    out_dir = "multi_ship_mc_outputs/plots"
    make_multi_ship_plots(csv_path, out_dir)
