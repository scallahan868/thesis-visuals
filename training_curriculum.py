# 21195 -- CTDE MAPPO
# 963fe -- GNN-DRL

# import wandb
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# # Choose the metric
# # Single-metric plots (3 of them)
# single_metrics = [
#     ("env_runners/custom_metrics/env/number_of_agents_mean", "Number of Agents (Mean)"),
#     (
#         "env_runners/custom_metrics/env/number_of_airports_mean",
#         "Number of Airports (Mean)",
#     ),
#     ("env_runners/custom_metrics/env/malfunction_rate_mean", "Malfunction Rate (Mean)"),
# ]

# # The two cargo metrics to combine into one subplot
# cargo_metrics = [
#     ("env_runners/custom_metrics/env/number_of_initial_cargo_mean", "Initial Cargo"),
#     ("env_runners/custom_metrics/dynamic_cargo_generated_mean", "Dynamic Cargo"),
# ]


# # Connect to wandb API
# api = wandb.Api()

# # Load CTDE MAPPO run
# run1 = api.run("socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000")

# # Load GNN-DRL run
# run2 = api.run("socallahan-air-force-institute-of-technology/new_thesis/909be_00000")


# def fetch_full(run, metric):
#     rows = []
#     for row in run.scan_history(keys=["training_iteration", metric]):
#         if row.get("training_iteration") is None or row.get(metric) is None:
#             continue
#         rows.append(
#             {
#                 "training_iteration": row["training_iteration"],
#                 "value": row[metric],
#             }
#         )

#     df = pd.DataFrame(rows)

#     df = (
#         df.sort_values("training_iteration")
#         .groupby("training_iteration", as_index=False)["value"]
#         .mean()
#     )
#     return df


# # --- layout: 2x2 (4 total plots) ---
# # --- layout: 2x2 (4 total plots) ---
# fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
# axes = axes.flatten()

# cargo_ax = axes[3]  # dedicate axis for cargo plot

# # 1) Plot the three single metrics
# for plot_ax, (metric_key, metric_label) in zip(axes[:3], single_metrics):
#     df_ctde = fetch_full(run1, metric_key)
#     df_gnn = fetch_full(run2, metric_key)

#     plot_ax.plot(
#         df_ctde["training_iteration"],
#         df_ctde["value"],
#         label="CTDE MAPPO",
#         color="tab:blue",
#         linewidth=3,
#     )
#     plot_ax.plot(
#         df_gnn["training_iteration"],
#         df_gnn["value"],
#         label="GNN-DRL",
#         color="tab:orange",
#         linewidth=3,
#     )

#     plot_ax.set_title(metric_label, fontsize=18)
#     plot_ax.set_ylabel(metric_label, fontsize=16)
#     plot_ax.grid(True)
#     plot_ax.legend(loc="lower right")

# # 2) Combined cargo plot (4 legend items)
# for metric_key, short_label in cargo_metrics:

#     df_ctde = fetch_full(run1, metric_key)
#     df_gnn = fetch_full(run2, metric_key)

#     is_dynamic = "dynamic" in metric_key.lower()
#     linestyle = "--" if is_dynamic else "-"

#     cargo_ax.plot(
#         df_ctde["training_iteration"],
#         df_ctde["value"],
#         label=f"CTDE MAPPO – {short_label}",
#         color="tab:blue",
#         linestyle=linestyle,
#         linewidth=3,
#     )

#     cargo_ax.plot(
#         df_gnn["training_iteration"],
#         df_gnn["value"],
#         label=f"GNN-DRL – {short_label}",
#         color="tab:orange",
#         linestyle=linestyle,
#         linewidth=3,
#     )

# cargo_ax.set_title("Cargo (Mean)", fontsize=18)
# cargo_ax.set_ylabel("Cargo (Mean)", fontsize=16)
# cargo_ax.grid(True)
# cargo_ax.legend(loc="best")

# # common x labels
# for plot_ax in axes:
#     plot_ax.set_xlabel("Training Iteration", fontsize=16)

# # put legend only on the cargo subplot (so it shows 4 items)
# cargo_ax.legend(loc="best", fontsize=9)

# fig.suptitle("Curriculum Training Metrics vs Training Iteration", fontsize=20)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# fig.savefig("figs/curriculum_training_metrics.png", dpi=300, bbox_inches="tight")
# plt.close(fig)


# training_curriculum.py (REWORKED)
# CTDE: 34c3d_00000
# GNN : 909be_00000

import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# ---- metrics ----
single_metrics = [
    ("env_runners/custom_metrics/env/number_of_agents_mean", "Number of Agents (Mean)"),
    (
        "env_runners/custom_metrics/env/number_of_airports_mean",
        "Number of Airports (Mean)",
    ),
    ("env_runners/custom_metrics/env/malfunction_rate_mean", "Malfunction Rate (Mean)"),
]

cargo_metrics = [
    ("env_runners/custom_metrics/env/number_of_initial_cargo_mean", "Initial Cargo"),
    ("env_runners/custom_metrics/dynamic_cargo_generated_mean", "Dynamic Cargo"),
]

ALL_METRICS = single_metrics + cargo_metrics

RUNS = [
    {
        "label": "CTDE MAPPO",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/34c3d_00000",
        "csv_out": "csv/ctde_curriculum.csv",
        "max_iterations": 2646,
    },
    {
        "label": "GNN-DRL",
        "path": "socallahan-air-force-institute-of-technology/new_thesis/e72f3_00000",
        "csv_out": "csv/gnn_curriculum.csv",
        "max_iterations": 1363,
    },
]

FIG_OUT = "figs/curriculum_training_metrics.png"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _series_last_nonnull_per_iter(run, metric_key: str) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      training_iteration, value
    where value is the LAST non-null metric observed for that iteration.
    """
    rows = []
    for row in run.scan_history(keys=["training_iteration", metric_key]):
        ti = row.get("training_iteration")
        if ti is None:
            continue
        rows.append({"training_iteration": int(ti), "value": row.get(metric_key)})

    if not rows:
        # No iteration rows at all (very unlikely)
        return pd.DataFrame(columns=["training_iteration", "value"])

    df = pd.DataFrame(rows).sort_values("training_iteration")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # take last non-null per iteration; if all null in that iter -> NaN (we'll fill later)
    df = df.groupby("training_iteration", as_index=False)["value"].agg(
        lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan
    )
    return df


def build_dense_metrics_df(run, metrics) -> pd.DataFrame:
    """
    Builds a single dense dataframe:
      training_iteration, <col per metric_key>
    with no NaNs (ffill/bfill).
    """
    # Gather per-metric series and determine max_iter across them
    per_metric = {}
    max_iter = 0

    for metric_key, _label in metrics:
        s = _series_last_nonnull_per_iter(run, metric_key)
        per_metric[metric_key] = s
        if not s.empty:
            max_iter = max(max_iter, int(s["training_iteration"].max()))

    if max_iter == 0:
        raise RuntimeError(f"No training_iteration rows found in run: {run.path}")

    base = pd.DataFrame({"training_iteration": range(1, max_iter + 1)})

    # Merge each metric into base
    out = base
    for metric_key, _label in metrics:
        s = per_metric[metric_key]
        col = metric_key.split("/")[-1]  # short-ish column name
        if s.empty:
            out[col] = np.nan
        else:
            out = out.merge(
                s.rename(columns={"value": col}),
                on="training_iteration",
                how="left",
            )

        # fill gaps for this column only
        out[col] = pd.to_numeric(out[col], errors="coerce").ffill().bfill()

        if out[col].isna().any():
            raise RuntimeError(
                f"Metric '{metric_key}' never appeared with a numeric value for run {run.path}."
            )

    return out


def main():
    api = wandb.Api()

    model_dfs = []
    for cfg in RUNS:
        run = api.run(cfg["path"])
        df = build_dense_metrics_df(run, ALL_METRICS)

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

    # ---- plotting (2x2) ----
    ensure_parent_dir(FIG_OUT)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    cargo_ax = axes[3]

    # three single metrics
    for i, (plot_ax, (metric_key, metric_label)) in enumerate(
        zip(axes[:3], single_metrics)
    ):
        col = metric_key.split("/")[-1]
        for label, df in model_dfs:
            plot_ax.plot(df["training_iteration"], df[col], label=label, linewidth=3)

        plot_ax.set_title(metric_label, fontsize=18)
        plot_ax.set_ylabel(metric_label, fontsize=16)
        plot_ax.grid(True)

        # Malfunction Rate (Mean) is at index 2, use 'upper left'
        legend_loc = "upper left" if i == 2 else "lower right"
        plot_ax.legend(loc=legend_loc, fontsize=14)

    # cargo combined (4 lines)
    # Color mapping for models
    color_map = {"CTDE MAPPO": "tab:blue", "GNN-DRL": "tab:orange"}

    for metric_key, short_label in cargo_metrics:
        col = metric_key.split("/")[-1]
        is_dynamic = "dynamic" in metric_key.lower()
        linestyle = "--" if is_dynamic else "-"

        for label, df in model_dfs:
            cargo_ax.plot(
                df["training_iteration"],
                df[col],
                label=f"{label} – {short_label}",
                color=color_map[label],
                linestyle=linestyle,
                linewidth=3,
            )

    cargo_ax.set_title("Cargo (Mean)", fontsize=18)
    cargo_ax.set_ylabel("Cargo (Mean)", fontsize=16)
    cargo_ax.grid(True)
    cargo_ax.legend(loc="best", fontsize=11)

    for ax in axes:
        ax.set_xlabel("Training Iteration", fontsize=16)

    fig.suptitle("Curriculum Training Metrics vs Training Iteration", fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {FIG_OUT}")


if __name__ == "__main__":
    main()
