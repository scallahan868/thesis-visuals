# # 21195 -- CTDE MAPPO
# # 963fe -- GNN-DRL

# import wandb
# import pandas as pd
# import matplotlib.pyplot as plt

# # Choose the metric
# metric = "timers/training_iteration_time_ms"

# # Connect to wandb API
# api = wandb.Api()

# # Load CTDE MAPPO run
# run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000")

# # Load GNN-DRL run
# run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/909be_00000")


# def fetch_full(run):
#     rows = []
#     for row in run.scan_history(keys=["training_iteration", metric]):
#         if row.get("training_iteration") is None or row.get(metric) is None:
#             continue
#         rows.append(
#             {
#                 "training_iteration": row["training_iteration"],
#                 "training_iteration_time": row[metric] / 1000.0,
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


# df_ctde = fetch_full(run1)
# df_gnn = fetch_full(run2)

# df_ctde.to_csv("csv/ctde_time.csv", index=False)
# df_gnn.to_csv("csv/gnn_time.csv", index=False)

# print("CTDE rows:", len(df_ctde), "iter max:", df_ctde["training_iteration"].max())
# print("GNN  rows:", len(df_gnn), "iter max:", df_gnn["training_iteration"].max())

# # Make the plot

# # ctde = pd.read_csv("csv/ctde_missed.csv")
# # gnn  = pd.read_csv("csv/gnn_missed.csv")

# plt.figure(figsize=(12, 6))
# plt.plot(
#     df_ctde["training_iteration"],
#     df_ctde["training_iteration_time"],
#     label="CTDE MAPPO",
# )
# plt.plot(
#     df_gnn["training_iteration"], df_gnn["training_iteration_time"], label="GNN-DRL"
# )
# plt.title("Training Iteration Time vs Training Iteration")
# plt.xlabel("Training Iteration")
# plt.ylabel("Training Iteration Time (s)")
# plt.grid(True)
# plt.legend()
# # plt.show()

# plt.savefig("figs/training_iteration_time.png", dpi=300, bbox_inches="tight")

# training_iteration_time.py (REWORKED)
# CTDE: 34c3d_00000
# GNN : 909be_00000

# training_iteration_time.py (REWORKED - SEPARATE PLOTS)
# CTDE: 34c3d_00000
# GNN : e72f3_00000  (change back to 909be_00000 if needed)

import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

TIME_METRIC = "timers/training_iteration_time_ms"

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

FIG_ITER_OUT = "figs/training_iteration_time_seconds_curr.png"
FIG_CUM_OUT = "figs/training_cumulative_time_hours_curr.png"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def fetch_dense_time(run) -> pd.DataFrame:
    rows = []
    for row in run.scan_history(keys=["training_iteration", TIME_METRIC]):
        ti = row.get("training_iteration")
        if ti is None:
            continue
        rows.append({"training_iteration": int(ti), "time_ms": row.get(TIME_METRIC)})

    if not rows:
        raise RuntimeError(f"No training_iteration rows found in run: {run.path}")

    df = pd.DataFrame(rows).sort_values("training_iteration")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")

    # last non-null time per iteration
    df = df.groupby("training_iteration", as_index=False)["time_ms"].agg(
        lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan
    )

    max_iter = int(df["training_iteration"].max())
    full = pd.DataFrame({"training_iteration": range(1, max_iter + 1)})
    df = full.merge(df, on="training_iteration", how="left").sort_values(
        "training_iteration"
    )

    # fill so there are no NaNs
    df["time_ms"] = df["time_ms"].ffill().bfill()
    if df["time_ms"].isna().any():
        raise RuntimeError(
            f"'{TIME_METRIC}' never appeared with a numeric value for run {run.path}."
        )

    df["training_iteration_time_s"] = df["time_ms"] / 1000.0
    df["cumulative_training_hours"] = df["training_iteration_time_s"].cumsum() / 3600.0

    return df[
        ["training_iteration", "training_iteration_time_s", "cumulative_training_hours"]
    ]


def plot_iteration_time(series):
    ensure_parent_dir(FIG_ITER_OUT)

    plt.figure(figsize=(12, 6))
    for label, df in series:
        plt.plot(
            df["training_iteration"],
            df["training_iteration_time_s"],
            label=label,
            linewidth=3,
        )

    plt.title("Training Iteration Time vs Training Iteration", fontsize=20)
    plt.xlabel("Training Iteration", fontsize=16)
    plt.ylabel("Training Iteration Time (seconds)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig(FIG_ITER_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_ITER_OUT}")


def plot_cumulative_time(series):
    ensure_parent_dir(FIG_CUM_OUT)

    plt.figure(figsize=(12, 6))

    for label, df in series:
        x = df["training_iteration"]
        y = df["cumulative_training_hours"]

        plt.plot(x, y, label=label, linewidth=3)

        # --- mark final point ---
        final_iter = x.iloc[-1]
        final_time = y.iloc[-1]

        plt.scatter([final_iter], [final_time], zorder=5)

        plt.annotate(
            f"{final_time:.2f} h",
            xy=(final_iter, final_time),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("Cumulative Training Time vs Training Iteration", fontsize=20)
    plt.xlabel("Training Iteration", fontsize=16)
    plt.ylabel("Cumulative Training Time (hours)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.savefig(FIG_CUM_OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {FIG_CUM_OUT}")


def main():
    api = wandb.Api()
    series = []

    for cfg in RUNS:
        run = api.run(cfg["path"])
        df = fetch_dense_time(run)

        # Limit to max_iterations
        max_iter = cfg.get("max_iterations")
        if max_iter is not None:
            df = df[df["training_iteration"] <= max_iter].reset_index(drop=True)

        ensure_parent_dir(cfg["csv_out"])
        df.to_csv(cfg["csv_out"], index=False)
        print(
            f"{cfg['label']} -> {cfg['csv_out']} "
            f"(rows={len(df)}, max_iter={df['training_iteration'].max()})"
        )

        series.append((cfg["label"], df))

    # two separate figures
    plot_iteration_time(series)
    plot_cumulative_time(series)


if __name__ == "__main__":
    main()
