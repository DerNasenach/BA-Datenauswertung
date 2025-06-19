import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def visualize_force_percent_differences(ax, json_path, title):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_key = next(iter(data))
    all_total_key = "all subjects"
    all_total = data[top_key][all_total_key]
    ohne = all_total["ohne exo"]
    mit = all_total["mit exo"]

    metrics = [
        ("mean of mean", "Mean of Means"),
        ("mean of max", "Mean of Max"),
    ]

    baseline = []
    relative = []
    labels = []

    for key, label in metrics:
        ohne_val = ohne[key]
        mit_val = mit[key]
        baseline.append(100)
        relative.append(100 * mit_val / ohne_val if ohne_val != 0 else 0)
        labels.append(label)

    x = range(len(labels))
    width = 0.35

    bars1 = ax.bar(x, baseline, width, label="ohne exo (100%)", color="#4F81BD")
    bars2 = ax.bar(
        [i + width for i in x],
        relative,
        width,
        label="mit exo (relative %)",
        color="#C0504D",
    )
    # Annotate mit exo bars with percentage
    for idx, bar in enumerate(bars2):
        height = bar.get_height()
        ax.annotate(
            f"{round(height)}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#C0504D",
            fontweight="bold",
        )

    ax.set_ylabel("Relative Value (%)")
    ax.set_title(title)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline + relative) * 1.2)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    return bars1, bars2


# Prepare file paths and titles
base_path = "Data/BKMP/evaluations"
params = [
    [os.path.join(base_path, "evaluations_concat.json"), "All Exercises"],
]
params += [
    [os.path.join(base_path, f"evaluations_exercise_{n}.json"), f"Exercise {n}"]
    for n in range(1, 7)
]

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

# First row: All Exercises (spanning all columns)
ax_all = fig.add_subplot(gs[0, :])
bars1, bars2 = visualize_force_percent_differences(ax_all, params[0][0], params[0][1])
all_bars = (bars1, bars2)

# Next 6: Exercises 1-6 in 2 rows, 3 columns each
axes = []
for i in range(6):
    row = 1 + i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    visualize_force_percent_differences(ax, params[i + 1][0], params[i + 1][1])
    axes.append(ax)

fig.legend(
    [all_bars[0], all_bars[1]],
    ["ohne exo (100%)", "mit exo (relative %)"],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize="large",
)
fig.suptitle(
    "BKMP: Relative Mean of Means and Mean of Max (mit exo vs ohne exo)",
    fontsize=16,
    y=1.06,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(
    os.path.join(base_path, "mean_and_max_overview.png"), dpi=300, bbox_inches="tight"
)

plt.show()
