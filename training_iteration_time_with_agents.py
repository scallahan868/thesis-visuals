# # 21195 -- CTDE MAPPO
# # 963fe -- GNN-DRL

# import wandb
# import pandas as pd
# import matplotlib.pyplot as plt

# # Choose the metrics
# time_metric = "timers/training_iteration_time_ms"
# agents_metric = "env_runners/custom_metrics/env/number_of_agents_mean"

# # Connect to wandb API
# api = wandb.Api()

# # Load CTDE MAPPO run
# run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000")

# # Load GNN-DRL run
# run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/909be_00000")


# def fetch_time_data(run):
#     rows = []
#     for row in run.scan_history(keys=["training_iteration", time_metric]):
#         if row.get("training_iteration") is None or row.get(time_metric) is None:
#             continue
#         rows.append(
#             {
#                 "training_iteration": row["training_iteration"],
#                 "training_iteration_time": row[time_metric] / 1000.0,
#             }
#         )
#     df = pd.DataFrame(rows)

#     # sort + de-dup (VERY important)
#     df = (
#         df.sort_values("training_iteration")
#         .groupby("training_iteration", as_index=False)["training_iteration_time"]
#         .mean()
#     )
#     return df


# def fetch_agents_data(run):
#     rows = []
#     for row in run.scan_history(keys=["training_iteration", agents_metric]):
#         if row.get("training_iteration") is None or row.get(agents_metric) is None:
#             continue
#         rows.append(
#             {
#                 "training_iteration": row["training_iteration"],
#                 "number_of_agents": row[agents_metric],
#             }
#         )
#     df = pd.DataFrame(rows)

#     # sort + de-dup (VERY important)
#     df = (
#         df.sort_values("training_iteration")
#         .groupby("training_iteration", as_index=False)["number_of_agents"]
#         .mean()
#     )
#     return df


# df_ctde_time = fetch_time_data(run1)
# df_gnn_time = fetch_time_data(run2)

# df_ctde_agents = fetch_agents_data(run1)
# df_gnn_agents = fetch_agents_data(run2)

# df_ctde_time.to_csv("csv/ctde_time.csv", index=False)
# df_gnn_time.to_csv("csv/gnn_time.csv", index=False)

# print(
#     "CTDE time rows:",
#     len(df_ctde_time),
#     "iter max:",
#     df_ctde_time["training_iteration"].max(),
# )
# print(
#     "GNN  time rows:",
#     len(df_gnn_time),
#     "iter max:",
#     df_gnn_time["training_iteration"].max(),
# )
# print(
#     "CTDE agents rows:",
#     len(df_ctde_agents),
#     "iter max:",
#     df_ctde_agents["training_iteration"].max(),
# )
# print(
#     "GNN  agents rows:",
#     len(df_gnn_agents),
#     "iter max:",
#     df_gnn_agents["training_iteration"].max(),
# )

# # Make the plot with dual y-axes

# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Plot training iteration time on the left y-axis
# ax1.set_xlabel("Training Iteration")
# ax1.set_ylabel("Training Iteration Time (s)")
# ax1.plot(
#     df_ctde_time["training_iteration"],
#     df_ctde_time["training_iteration_time"],
#     label="CTDE MAPPO - Time",
#     color="tab:blue",
#     linestyle="-",
# )
# ax1.plot(
#     df_gnn_time["training_iteration"],
#     df_gnn_time["training_iteration_time"],
#     label="GNN-DRL - Time",
#     color="tab:orange",
#     linestyle="-",
# )
# ax1.tick_params(axis="y")
# ax1.grid(True)

# # Create a second y-axis on the right for number of agents
# ax2 = ax1.twinx()
# ax2.set_ylabel("Number of Agents (Mean)")
# ax2.plot(
#     df_ctde_agents["training_iteration"],
#     df_ctde_agents["number_of_agents"],
#     label="CTDE MAPPO - Agents",
#     color="tab:blue",
#     linestyle="--",
# )
# ax2.plot(
#     df_gnn_agents["training_iteration"],
#     df_gnn_agents["number_of_agents"],
#     label="GNN-DRL - Agents",
#     color="tab:orange",
#     linestyle="--",
# )
# ax2.tick_params(axis="y")

# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

# plt.title("Training Iteration Time and Number of Agents vs Training Iteration")
# fig.tight_layout()

# plt.savefig(
#     "figs/training_iteration_time_with_agents.png", dpi=300, bbox_inches="tight"
# )


# training_iteration_time_with_agents.py (REWORKED)
# CTDE: 34c3d_00000
# GNN : 909be_00000

import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

TIME_METRIC = "timers/training_iteration_time_ms"
AGENTS_METRIC = "env_runners/custom_metrics/env/number_of_agents_mean"

RUNS = [
    {
        "label": "CTDE MAPPO",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000",
        "csv_out": "csv/ctde_time_with_agents.csv",
        "max_iterations": 2646,
    },
    {
        "label": "GNN-DRL",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/e72f3_00000",
        "csv_out": "csv/gnn_time_with_agents.csv",
        "max_iterations": 1363,
    },
]

FIG_OUT = "figs/training_iteration_time_with_agents.png"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _dense_last_nonnull(run, metric_key: str, out_col: str) -> pd.DataFrame:
    rows = []
    for row in run.scan_history(keys=["training_iteration", metric_key]):
        ti = row.get("training_iteration")
        if ti is None:
            continue
        rows.append({"training_iteration": int(ti), out_col: row.get(metric_key)})

    if not rows:
        return pd.DataFrame(columns=["training_iteration", out_col])

    df = pd.DataFrame(rows).sort_values("training_iteration")
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce")

    df = df.groupby("training_iteration", as_index=False)[out_col].agg(
        lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan
    )
    return df


def build_dense_time_agents(run) -> pd.DataFrame:
    tdf = _dense_last_nonnull(run, TIME_METRIC, "time_ms")
    adf = _dense_last_nonnull(run, AGENTS_METRIC, "number_of_agents")

    if tdf.empty and adf.empty:
        raise RuntimeError(f"No training_iteration rows found in run: {run.path}")

    max_iter = 0
    if not tdf.empty:
        max_iter = max(max_iter, int(tdf["training_iteration"].max()))
    if not adf.empty:
        max_iter = max(max_iter, int(adf["training_iteration"].max()))

    base = pd.DataFrame({"training_iteration": range(1, max_iter + 1)})

    out = base.merge(tdf, on="training_iteration", how="left").merge(
        adf, on="training_iteration", how="left"
    )
    out["time_ms"] = out["time_ms"].ffill().bfill()
    out["number_of_agents"] = out["number_of_agents"].ffill().bfill()

    if out["time_ms"].isna().any():
        raise RuntimeError(
            f"'{TIME_METRIC}' never appeared with a numeric value for run {run.path}."
        )
    if out["number_of_agents"].isna().any():
        raise RuntimeError(
            f"'{AGENTS_METRIC}' never appeared with a numeric value for run {run.path}."
        )

    out["training_iteration_time_s"] = out["time_ms"] / 1000.0
    out["cumulative_training_hours"] = (
        out["training_iteration_time_s"].cumsum() / 3600.0
    )

    return out[
        [
            "training_iteration",
            "training_iteration_time_s",
            "cumulative_training_hours",
            "number_of_agents",
        ]
    ]


def main():
    api = wandb.Api()
    model_dfs = []

    for cfg in RUNS:
        run = api.run(cfg["path"])
        df = build_dense_time_agents(run)

        # Limit to max_iterations
        max_iter = cfg.get("max_iterations")
        if max_iter is not None:
            df = df[df["training_iteration"] <= max_iter].reset_index(drop=True)

        ensure_parent_dir(cfg["csv_out"])
        df.to_csv(cfg["csv_out"], index=False)
        print(
            f"{cfg['label']} -> {cfg['csv_out']} (rows={len(df)}, max_iter={df['training_iteration'].max()})"
        )

        model_dfs.append((cfg["label"], df))

    ensure_parent_dir(FIG_OUT)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Training Iteration Time (s)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of Agents (Mean)")

    for label, df in model_dfs:
        ax1.plot(
            df["training_iteration"],
            df["training_iteration_time_s"],
            label=f"{label} – Time",
            linestyle="-",
            linewidth=3,
        )
        ax2.plot(
            df["training_iteration"],
            df["number_of_agents"],
            label=f"{label} – Agents",
            linestyle="--",
            linewidth=3,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=16)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=16)
    ax1.set_xlabel("Training Iteration", fontsize=16)

    plt.title(
        "Training Iteration Time and Number of Agents vs Training Iteration",
        fontsize=20,
    )
    fig.tight_layout()
    plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {FIG_OUT}")


if __name__ == "__main__":
    main()
