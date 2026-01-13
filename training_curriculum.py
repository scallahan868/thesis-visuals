# 21195 -- CTDE MAPPO
# 963fe -- GNN-DRL

import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Choose the metric
metrics = [
    ("env_runners/custom_metrics/env/number_of_agents_mean",
     "Number of Agents (Mean)"),

    ("env_runners/custom_metrics/env/number_of_airports_mean",
     "Number of Airports (Mean)"),

    ("env_runners/custom_metrics/env/number_of_initial_cargo_mean",
     "Number of Initial Cargo (Mean)"),

    ("env_runners/custom_metrics/dynamic_cargo_generated_mean",
     "Dynamic Cargo Generated (Mean)"),

     ("env_runners/custom_metrics/env/malfunction_rate_mean",
     "Malfunction Rate (Mean)"),
]

# Connect to wandb API
api = wandb.Api()

# Load CTDE MAPPO run
run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/21195_00000")

# Load GNN-DRL run
run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/963fe_00000")

def fetch_full(run, metric):
    rows = []
    for row in run.scan_history(keys=["training_iteration", metric]):
        if row.get("training_iteration") is None or row.get(metric) is None:
            continue
        rows.append({
            "training_iteration": row["training_iteration"],
            "value": row[metric],
        })

    df = pd.DataFrame(rows)

    df = (
        df.sort_values("training_iteration")
          .groupby("training_iteration", as_index=False)["value"]
          .mean()
    )
    return df

fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
axes = axes.flatten()

for ax, (metric_key, metric_label) in zip(axes, metrics):

    df_ctde = fetch_full(run1, metric_key)
    df_gnn  = fetch_full(run2, metric_key)

    ax.plot(df_ctde["training_iteration"], df_ctde["value"], label="CTDE MAPPO")
    ax.plot(df_gnn["training_iteration"],  df_gnn["value"],  label="GNN-DRL")

    ax.set_title(metric_label)
    ax.set_ylabel(metric_label)
    ax.grid(True)

df_ctde.to_csv("csv/ctde_curriculum.csv", index=False)
df_gnn.to_csv("csv/gnn_curriculum.csv", index=False)

# Make the plot

# Common X label
for ax in axes:
    ax.set_xlabel("Training Iteration")

# Single legend for entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)

fig.suptitle("Curriculum Training Metrics vs Training Iteration", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# plt.show()

plt.savefig("figs/curriculum_training_metrics.png", dpi=300, bbox_inches="tight")
