# 21195 -- CTDE MAPPO
# 963fe -- GNN-DRL

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Choose the metric
# Single-metric plots (3 of them)
single_metrics = [
    ("env_runners/custom_metrics/env/number_of_agents_mean",  "Number of Agents (Mean)"),
    ("env_runners/custom_metrics/env/number_of_airports_mean","Number of Airports (Mean)"),
    ("env_runners/custom_metrics/env/malfunction_rate_mean",  "Malfunction Rate (Mean)"),
]

# The two cargo metrics to combine into one subplot
cargo_metrics = [
    ("env_runners/custom_metrics/env/number_of_initial_cargo_mean", "Initial Cargo"),
    ("env_runners/custom_metrics/dynamic_cargo_generated_mean",     "Dynamic Cargo"),
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

# --- layout: 2x2 (4 total plots) ---
# --- layout: 2x2 (4 total plots) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

cargo_ax = axes[3]   # dedicate axis for cargo plot

# 1) Plot the three single metrics
for plot_ax, (metric_key, metric_label) in zip(axes[:3], single_metrics):
    df_ctde = fetch_full(run1, metric_key)
    df_gnn  = fetch_full(run2, metric_key)

    plot_ax.plot(df_ctde["training_iteration"], df_ctde["value"],
                 label="CTDE MAPPO", color="tab:blue")
    plot_ax.plot(df_gnn["training_iteration"], df_gnn["value"],
                 label="GNN-DRL", color="tab:orange")

    plot_ax.set_title(metric_label)
    plot_ax.set_ylabel(metric_label)
    plot_ax.grid(True)
    plot_ax.legend(loc="lower right")

# 2) Combined cargo plot (4 legend items)
for metric_key, short_label in cargo_metrics:

    df_ctde = fetch_full(run1, metric_key)
    df_gnn  = fetch_full(run2, metric_key)

    is_dynamic = "dynamic" in metric_key.lower()
    linestyle = "--" if is_dynamic else "-"

    cargo_ax.plot(
        df_ctde["training_iteration"], df_ctde["value"],
        label=f"CTDE MAPPO – {short_label}",
        color="tab:blue",
        linestyle=linestyle
    )

    cargo_ax.plot(
        df_gnn["training_iteration"], df_gnn["value"],
        label=f"GNN-DRL – {short_label}",
        color="tab:orange",
        linestyle=linestyle
    )

cargo_ax.set_title("Cargo (Mean)")
cargo_ax.set_ylabel("Cargo (Mean)")
cargo_ax.grid(True)
cargo_ax.legend(loc="best")

# common x labels
for plot_ax in axes:
    plot_ax.set_xlabel("Training Iteration")

# put legend only on the cargo subplot (so it shows 4 items)
cargo_ax.legend(loc="best")

fig.suptitle("Curriculum Training Metrics vs Training Iteration", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

fig.savefig("figs/curriculum_training_metrics.png", dpi=300, bbox_inches="tight")
plt.close(fig)

