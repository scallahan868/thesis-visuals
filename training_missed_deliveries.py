# 21195 -- CTDE MAPPO
# 963fe -- GNN-DRL

import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Choose the metric
metric = "env_runners/custom_metrics/proportion_deliveries_missed_mean"

# Connect to wandb API
api = wandb.Api()

# Load CTDE MAPPO run
run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/21195_00000")

# Load GNN-DRL run
run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/963fe_00000")

def fetch_full(run):
    rows = []
    for row in run.scan_history(keys=["training_iteration", metric]):
        if row.get("training_iteration") is None or row.get(metric) is None:
            continue
        rows.append({
            "training_iteration": row["training_iteration"],
            "proportion_missed": row[metric],
        })
    df = pd.DataFrame(rows)

    # sort + de-dup (VERY important)
    df = (df.sort_values("training_iteration")
            .groupby("training_iteration", as_index=False)["proportion_missed"].mean())
    return df

df_ctde = fetch_full(run1)
df_gnn  = fetch_full(run2)

df_ctde.to_csv("csv/ctde_missed.csv", index=False)
df_gnn.to_csv("csv/gnn_missed.csv", index=False)

print("CTDE rows:", len(df_ctde), "iter max:", df_ctde["training_iteration"].max())
print("GNN  rows:", len(df_gnn),  "iter max:", df_gnn["training_iteration"].max())

# Make the plot

# ctde = pd.read_csv("csv/ctde_missed.csv")
# gnn  = pd.read_csv("csv/gnn_missed.csv")

plt.figure(figsize=(12,6))
plt.plot(df_ctde["training_iteration"], df_ctde["proportion_missed"], label="CTDE MAPPO")
plt.plot(df_gnn["training_iteration"],  df_gnn["proportion_missed"],  label="GNN-DRL")
plt.title("Proportion of Deliveries Missed vs Training Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Proportion of Deliveries Missed (Mean)")
plt.grid(True)
plt.axhline(y=0.3, linestyle="--", linewidth=2, label="Target = 0.30", color="red")
plt.legend()
# plt.show()

plt.savefig("figs/proportion_deliveries_missed.png", dpi=300, bbox_inches="tight")
