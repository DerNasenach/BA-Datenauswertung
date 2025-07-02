import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_angle_percent_differences(ax, json_path, name_layer_1, value_type="mean"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_total = data[name_layer_1]["all subjects"]
    ohne = all_total["ohne exo"]
    mit = all_total["mit exo"]
    prefix = "mean_"

    if value_type == "mean":
        metrics = [
            ("mean_torsion", "Mean Torsion"),
            ("mean_flexion", "Mean Flexion"),
            ("mean_extension", "Mean Extension"),
            ("mean_total_angle", "Mean Total Angle"),
        ]
    elif value_type == "max":
        metrics = [
            ("max_torsion", "Max Torsion"),
            ("max_flexion", "Max Flexion"),
            ("max_extension", "Max Extension"),
            ("max_total_angle", "Max Total Angle"),
        ]

    baseline = []
    relative = []
    labels = []

    for key, label in metrics:
        ohne_val = ohne[f"{prefix}{key}"]
        mit_val = mit[f"{prefix}{key}"]
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
    ax.set_title(name_layer_1)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline + relative) * 1.2)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    return bars1, bars2


def visualize_angle_absolute_differences(
    ax, json_path, name_layer_1, value_type="mean"
):
    import numpy as np

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_total = data[name_layer_1]["all subjects"]
    ohne = all_total["ohne exo"]
    mit = all_total["mit exo"]
    prefix = "mean_"

    if value_type == "mean":
        metrics = [
            ("mean_torsion", "Mean Torsion"),
            ("mean_flexion", "Mean Flexion"),
            ("mean_extension", "Mean Extension"),
            ("mean_total_angle", "Mean Total Angle"),
        ]
        flexion_key = "mean_flexion"
        extension_key = "mean_extension"
    elif value_type == "max":
        metrics = [
            ("max_torsion", "Max Torsion"),
            ("max_flexion", "Max Flexion"),
            ("max_extension", "Max Extension"),
            ("max_total_angle", "Max Total Angle"),
        ]
        flexion_key = "max_flexion"
        extension_key = "max_extension"

    ohne_vals = []
    mit_vals = []
    labels = []
    percent_diffs = []

    # Add the original metrics
    for key, label in metrics:
        ohne_val = ohne[f"{prefix}{key}"]
        mit_val = mit[f"{prefix}{key}"]
        ohne_vals.append(ohne_val)
        mit_vals.append(mit_val)
        labels.append(label)
        percent_diff = 100 * (mit_val - ohne_val) / ohne_val if ohne_val != 0 else 0
        percent_diffs.append(percent_diff)

    # Add Range of Motion (ROM)
    ohne_rom = ohne[f"{prefix}{flexion_key}"] + ohne[f"{prefix}{extension_key}"]
    mit_rom = mit[f"{prefix}{flexion_key}"] + mit[f"{prefix}{extension_key}"]
    ohne_vals.append(ohne_rom)
    mit_vals.append(mit_rom)
    labels.append("Mean Range of Motion")
    percent_diff_rom = 100 * (mit_rom - ohne_rom) / ohne_rom if ohne_rom != 0 else 0
    percent_diffs.append(percent_diff_rom)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x, ohne_vals, width, label="ohne exo", color="#4F81BD")
    bars2 = ax.bar(x + width, mit_vals, width, label="mit exo", color="#C0504D")

    for idx, bar in enumerate(bars2):
        height = bar.get_height()
        diff = percent_diffs[idx]
        ax.annotate(
            f"{diff:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#C0504D",
            fontweight="bold",
        )

    ax.set_ylabel("Angle in degrees")
    ax.set_title(name_layer_1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, max(ohne_vals + mit_vals) * 1.2)
    return bars1, bars2


params = [["Data/CAPTIV/evaluations/evaluations_concat.json", "all exercises"]] + [
    [f"Data/CAPTIV/evaluations/evaluations_exercise_{n}.json", f"exercise {n}"]
    for n in range(1, 7)
]

# plot mean angle
fig_mean = plt.figure(figsize=(18, 10))
gs_mean = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

ax_all_mean = fig_mean.add_subplot(gs_mean[0, :])
bars1_mean, bars2_mean = visualize_angle_absolute_differences(
    ax_all_mean, params[0][0], params[0][1], value_type="mean"
)
all_bars_mean = (bars1_mean, bars2_mean)

axes_mean = []
for i in range(6):
    row = 1 + i // 3
    col = i % 3
    ax = fig_mean.add_subplot(gs_mean[row, col])
    visualize_angle_absolute_differences(
        ax, params[i + 1][0], params[i + 1][1], value_type="mean"
    )
    axes_mean.append(ax)

fig_mean.legend(
    [all_bars_mean[0], all_bars_mean[1]],
    ["ohne exo (100%)", "mit exo (relative %)"],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize="large",
)
fig_mean.suptitle(
    "Relative Mean Angles: mit exo vs ohne exo (All Subjects & Exercises)",
    fontsize=16,
    y=1.06,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig_mean.savefig(
    "Data/CAPTIV/evaluations/mean_angles_overview.png", dpi=300, bbox_inches="tight"
)

# plot max angle
fig_max = plt.figure(figsize=(18, 10))
gs_max = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

ax_all_max = fig_max.add_subplot(gs_max[0, :])
bars1_max, bars2_max = visualize_angle_absolute_differences(
    ax_all_max, params[0][0], params[0][1], value_type="max"
)
all_bars_max = (bars1_max, bars2_max)

axes_max = []
for i in range(6):
    row = 1 + i // 3
    col = i % 3
    ax = fig_max.add_subplot(gs_max[row, col])
    visualize_angle_absolute_differences(
        ax, params[i + 1][0], params[i + 1][1], value_type="max"
    )
    axes_max.append(ax)

fig_max.legend(
    [all_bars_max[0], all_bars_max[1]],
    ["ohne exo (100%)", "mit exo (relative %)"],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize="large",
)
fig_max.suptitle(
    "Relative Max Angles: mit exo vs ohne exo (All Subjects & Exercises)",
    fontsize=16,
    y=1.06,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig_max.savefig(
    "Data/CAPTIV/evaluations/max_angles_overview.png", dpi=300, bbox_inches="tight"
)

plt.show()
