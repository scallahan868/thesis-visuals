# # 21195 -- CTDE MAPPO
# # 963fe -- GNN-DRL

# import wandb
# import pandas as pd
# import matplotlib.pyplot as plt

# # Choose the metric
# metric = "env_runners/custom_metrics/proportion_deliveries_missed_mean"

# # Connect to wandb API
# api = wandb.Api()

# # Load CTDE MAPPO run
# run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000")

# # Load GNN-DRL run
# run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/e72f3_00000")


# def fetch_full(run):
#     rows = []
#     for row in run.scan_history(keys=["training_iteration", metric]):
#         if row.get("training_iteration") is None or row.get(metric) is None:
#             continue
#         rows.append(
#             {
#                 "training_iteration": row["training_iteration"],
#                 "proportion_missed": row[metric],
#             }
#         )
#     df = pd.DataFrame(rows)

#     # sort + de-dup (VERY important)
#     df = (
#         df.sort_values("training_iteration")
#         .groupby("training_iteration", as_index=False)["proportion_missed"]
#         .mean()
#     )
#     return df


# df_ctde = fetch_full(run1)
# df_gnn = fetch_full(run2)

# df_ctde.to_csv("csv/ctde_missed.csv", index=False)
# df_gnn.to_csv("csv/gnn_missed.csv", index=False)

# print("CTDE rows:", len(df_ctde), "iter max:", df_ctde["training_iteration"].max())
# print("GNN  rows:", len(df_gnn), "iter max:", df_gnn["training_iteration"].max())

# # Make the plot

# # ctde = pd.read_csv("csv/ctde_missed.csv")
# # gnn  = pd.read_csv("csv/gnn_missed.csv")

# plt.figure(figsize=(12, 6))
# plt.plot(
#     df_ctde["training_iteration"], df_ctde["proportion_missed"], label="CTDE MAPPO"
# )
# plt.plot(df_gnn["training_iteration"], df_gnn["proportion_missed"], label="GNN-DRL")
# plt.title("Proportion of Deliveries Missed vs Training Iteration")
# plt.xlabel("Training Iteration")
# plt.ylabel("Proportion of Deliveries Missed (Mean)")
# plt.grid(True)
# plt.axhline(y=0.3, linestyle="--", linewidth=2, label="Target = 0.30", color="red")
# plt.legend()
# # plt.show()

# plt.savefig("figs/proportion_deliveries_missed.png", dpi=300, bbox_inches="tight")

# training_missed_deliveries_reworked.py
import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Config
# ------------------
METRIC = "env_runners/custom_metrics/proportion_deliveries_missed_mean"

RUNS = [
    {
        "label": "CTDE MAPPO - with curriculum",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000",
        "csv_out": "csv/ctde_missed.csv",
        "max_iterations": 2646,
    },
    {
        "label": "CTDE MAPPO - without curriculum",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/74b46_00000",
        "csv_out": "csv/gnn_missed.csv",
        "max_iterations": 2646,
    },
]

PLOT_OUT = "figs/proportion_deliveries_missed_curr.png"
TARGET = 0.30


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def fetch_complete_series(run: wandb.apis.public.Run, metric_key: str) -> pd.DataFrame:
    """
    Dense per-iteration dataframe with no NaNs.

    Strategy:
    1) scan_history over (training_iteration, metric)
    2) for each training_iteration, take the LAST non-null metric seen
    3) reindex to every iteration 1..max_iter
    4) forward-fill missing values; backfill any initial gap
    """
    rows = []
    for row in run.scan_history(keys=["training_iteration", metric_key]):
        ti = row.get("training_iteration")
        if ti is None:
            continue
        rows.append(
            {"training_iteration": int(ti), "proportion_missed": row.get(metric_key)}
        )

    if not rows:
        raise RuntimeError(
            f"No history rows found with training_iteration for run {run.path}."
        )

    df = pd.DataFrame(rows).sort_values("training_iteration")

    # Take the last non-null value per iteration (if multiple rows per iter)
    df["proportion_missed"] = pd.to_numeric(df["proportion_missed"], errors="coerce")
    df = df.groupby("training_iteration", as_index=False)["proportion_missed"].agg(
        lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan
    )

    max_iter = int(df["training_iteration"].max())

    # Force every iteration to exist
    full = pd.DataFrame({"training_iteration": range(1, max_iter + 1)})
    df = full.merge(df, on="training_iteration", how="left").sort_values(
        "training_iteration"
    )

    # Fill gaps so there are no NaNs
    df["proportion_missed"] = df["proportion_missed"].ffill().bfill()

    if df["proportion_missed"].isna().any():
        # This would mean the metric never appeared at all.
        raise RuntimeError(
            f"Metric '{metric_key}' never appeared in history for run {run.path}."
        )

    return df


def main() -> None:
    api = wandb.Api()

    series = []
    for cfg in RUNS:
        run = api.run(cfg["path"])
        df = fetch_complete_series(run, METRIC)

        # Limit to max_iterations
        max_iter = cfg.get("max_iterations")
        if max_iter is not None:
            df = df[df["training_iteration"] <= max_iter].reset_index(drop=True)

        ensure_parent_dir(cfg["csv_out"])
        df.to_csv(cfg["csv_out"], index=False)
        print(
            f"{cfg['label']}: rows={len(df)} max_iter={df['training_iteration'].max()} csv={cfg['csv_out']}"
        )

        series.append((cfg["label"], df))

    # Plot both on one chart
    ensure_parent_dir(PLOT_OUT)
    plt.figure(figsize=(12, 6))

    for label, df in series:
        plt.plot(
            df["training_iteration"], df["proportion_missed"], label=label, linewidth=2
        )

    plt.title("Proportion of Deliveries Missed vs Training Iteration", fontsize=20)
    plt.xlabel("Training Iteration", fontsize=16)
    plt.ylabel("Proportion of Deliveries Missed (Mean)", fontsize=16)
    plt.grid(True)
    plt.axhline(
        y=TARGET,
        linestyle="--",
        linewidth=3,
        label=f"Target = {TARGET:.2f}",
        color="red",
    )
    plt.legend(fontsize=14)

    plt.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {PLOT_OUT}")


if __name__ == "__main__":
    main()
