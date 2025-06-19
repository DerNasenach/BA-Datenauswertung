import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def visualize_emg_values(ax, json_path, title):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use "all subjects" for exercise files, "all subjects" for aggregate
    all_total = data["all subjects"]

    metrics = [
        ("max", "Max"),
        ("mean", "Mean"),
        ("median frequency", "Median Freq."),
    ]

    ohne = all_total["ohne exo"]
    mit = all_total["mit exo"]

    ohne_vals = [ohne[key] for key, _ in metrics]
    mit_vals = [mit[key] for key, _ in metrics]
    labels = [label for _, label in metrics]

    baseline = [100 for _ in ohne_vals]
    relative = [
        100 * mit_val / ohne_val if ohne_val != 0 else 0
        for mit_val, ohne_val in zip(mit_vals, ohne_vals)
    ]

    x = range(len(labels))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x], baseline, width, label="ohne exo", color="#4F81BD"
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], relative, width, label="mit exo", color="#C0504D"
    )

    # Annotate bars with values
    for bar, val in zip(bars2, relative):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=bar.get_facecolor(),
            fontweight="bold",
        )

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline + relative) * 1.2)
    return bars1, bars2


# List of all muscle filenames and pretty names
muscles = [
    ("Biceps_femoris_left.json", "Biceps femoris left"),
    ("Biceps_femoris_right.json", "Biceps femoris right"),
    ("Quadriceps_left.json", "Quadriceps left"),
    ("Quadriceps_right.json", "Quadriceps right"),
    ("Gluteus_maximus_left.json", "Gluteus maximus left"),
    ("Gluteus_maximus_right.json", "Gluteus maximus right"),
    ("Erector_spinae_left.json", "Erector spinae left"),
    ("Erector_spinae_right.json", "Erector spinae right"),
]

base_path = "Data/EMG/evaluations"

for muscle_file, muscle_name in muscles:
    params = [
        [os.path.join(base_path, "aggregate", muscle_file), "All Exercises"],
    ]
    params += [
        [os.path.join(base_path, f"Exercise {n}", muscle_file), f"Exercise {n}"]
        for n in range(1, 7)
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # First row: All Exercises (spanning all columns)
    ax_all = fig.add_subplot(gs[0, :])
    bars1, bars2 = visualize_emg_values(ax_all, params[0][0], params[0][1])
    all_bars = (bars1, bars2)

    # Next 6: Exercises 1-6 in 2 rows, 3 columns each
    axes = []
    for i in range(6):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        visualize_emg_values(ax, params[i + 1][0], params[i + 1][1])
        axes.append(ax)

    fig.legend(
        [all_bars[0], all_bars[1]],
        ["ohne exo", "mit exo"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        fontsize="large",
    )
    fig.suptitle(
        f"EMG: Max, Mean, and Median Frequency (ohne exo vs mit exo)\n{muscle_name}",
        fontsize=16,
        y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(
        os.path.join(base_path, f"{muscle_name.replace(' ', '_')}_overview.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


"""
# Prepare file paths and titles
base_path = "Data/EMG/evaluations"
muscle = "Biceps_femoris_left.json"
params = [
    [os.path.join(base_path, "aggregate", muscle), "All Exercises"],
]
params += [
    [os.path.join(base_path, f"Exercise {n}", muscle), f"Exercise {n}"]
    for n in range(1, 7)
]

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

# First row: All Exercises (spanning all columns)
ax_all = fig.add_subplot(gs[0, :])
bars1, bars2 = visualize_emg_values(ax_all, params[0][0], params[0][1])
all_bars = (bars1, bars2)

# Next 6: Exercises 1-6 in 2 rows, 3 columns each
axes = []
for i in range(6):
    row = 1 + i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    visualize_emg_values(ax, params[i + 1][0], params[i + 1][1])
    axes.append(ax)

fig.legend(
    [all_bars[0], all_bars[1]],
    ["ohne exo", "mit exo"],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize="large",
)
fig.suptitle(
    "EMG: Max, Mean, and Median Frequency (ohne exo vs mit exo)\nBiceps femoris left",
    fontsize=16,
    y=1.06,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(
    os.path.join(base_path, "Biceps_femoris_left_overview.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
"""
