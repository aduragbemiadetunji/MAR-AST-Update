
import pandas as pd

def summarize_violations_by_contract(csv_path):
    """
    Load a CSV file containing contract violations and return a pivot table 
    showing counts of violations per contract_id per subsystem.
    
    Parameters:
    - csv_path (str): Path to the violations CSV file
    
    Returns:
    - pd.DataFrame: Pivot table (subsystem x contract_id)
    """
    df = pd.read_csv(csv_path)

    # Group by subsystem and specific contract_id (A1, G1, etc.)
    detailed_agg = df.groupby(["subsystem", "contract_id"]).size().reset_index(name="Violations")

    # Add total per subsystem
    total_agg = df["subsystem"].value_counts().reset_index()
    total_agg.columns = ["subsystem", "Violations"]
    total_agg["contract_id"] = "TOTAL"

    # Combine detailed and totals
    combined_detailed = pd.concat([detailed_agg, total_agg], ignore_index=True)

    # Pivot for table format
    pivot_contracts = combined_detailed.pivot(index="subsystem", columns="contract_id", values="Violations").fillna(0).astype(int)

    return pivot_contracts



# pivot_table = summarize_violations_by_contract("contracts/logs/contract_violations.csv")
# pivot_table = summarize_violations_by_contract("mc_outputs\\violations_episode_001_seed_1667601611.csv")
pivot_table = summarize_violations_by_contract("multi_ship_mc_outputs\\violations_episode_001_seed_1667601611.csv")
print(pivot_table)
