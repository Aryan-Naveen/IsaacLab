import pandas as pd
import numpy as np

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "runs/eval/multi_eval_legtrain/all_rollouts.csv"

SUCCESS_THRESHOLD = 750

METHOD_RENAME = {
    "CTRLSAC-multi": "RepMT",
    "CTRLSAC-false": "CTRL",
    "CTRLSAC": "CTRL",
    "SAC": "SAC",
}

CHECKPOINT_MAP = {
    "best_agent.pt": "Best",
    "agent_250000.pt": "Last",
}

# ---------------------------
# Helper functions
# ---------------------------
def classify_task(task_name):
    """
    Customize this based on your naming convention.
    Example assumptions:
      - contains 'source'
      - contains 'in'
      - contains 'ood'
    """
    task_name = str(task_name).lower()
    if "source" in task_name:
        return "source"
    elif "ood" in task_name:
        return "ood"
    else:
        return "in_domain"


def compute_success(row):
    return (row["reward"] > SUCCESS_THRESHOLD) and (not row["crashed"])


# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(CSV_PATH)

# Normalize export schema (multi_eval rollouts) vs legacy column names
if "method" not in df.columns and "agent_type" in df.columns:
    df = df.rename(
        columns={
            "agent_type": "method",
            "ckpt": "checkpoint",
            "total_reward": "reward",
            "section": "task",
        }
    )
    seed_m = df["folder"].astype(str).str.extract(r"seed_(\d+)", expand=False)
    df["seed"] = pd.to_numeric(seed_m, errors="coerce")

# Expected columns:
# ['method', 'checkpoint', 'seed', 'task', 'reward', 'crashed']

# ---------------------------
# Preprocess
# ---------------------------
if "method" not in df.columns:
    raise KeyError(
        "CSV must include a 'method' column or 'agent_type' (multi_eval rollouts export)."
    )

df["method"] = df["method"].map(METHOD_RENAME).fillna(df["method"])
df["checkpoint"] = df["checkpoint"].map(CHECKPOINT_MAP)

df = df[df["checkpoint"].notna()]  # keep only best/last

df["task_type"] = df["task"].apply(classify_task)
df["success"] = df.apply(compute_success, axis=1)

# ---------------------------
# Aggregate per seed first
# ---------------------------
grouped = (
    df.groupby(["method", "checkpoint", "seed", "task_type"])
    .agg(
        avg_reward=("reward", "mean"),
        success_rate=("success", "mean"),
    )
    .reset_index()
)

# ---------------------------
# Then average across seeds
# ---------------------------
final = (
    grouped.groupby(["method", "checkpoint", "task_type"])
    .agg(
        reward_mean=("avg_reward", "mean"),
        reward_std=("avg_reward", "std"),
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
    )
    .reset_index()
)

# ---------------------------
# Pivot into nice table
# ---------------------------
def format_metric(mean, std):
    if np.isnan(std):
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {std:.1f}"


rows = []

for (method, checkpoint), subdf in final.groupby(["method", "checkpoint"]):
    row = {
        "Method": method,
        "Checkpoint": checkpoint,
    }

    for task_type in ["source", "in_domain", "ood"]:
        task_df = subdf[subdf["task_type"] == task_type]

        if len(task_df) == 0:
            row[f"{task_type}_reward"] = "-"
            row[f"{task_type}_success"] = "-"
            continue

        r_mean = task_df["reward_mean"].values[0]
        r_std = task_df["reward_std"].values[0]
        s_mean = task_df["success_mean"].values[0]
        s_std = task_df["success_std"].values[0]

        row[f"{task_type}_reward"] = format_metric(r_mean, r_std)
        row[f"{task_type}_success"] = format_metric(100 * s_mean, 100 * s_std)

    rows.append(row)

table_df = pd.DataFrame(rows)

# ---------------------------
# Pretty print
# ---------------------------
def print_table(df):
    cols = [
        "Method", "Checkpoint",
        "source_reward", "source_success",
        "in_domain_reward", "in_domain_success",
        "ood_reward", "ood_success"
    ]
    df = df[cols]

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


print_table(table_df)