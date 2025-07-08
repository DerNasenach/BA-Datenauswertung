import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np


def visualize_absolute_emg_values(ax, json_path, title):
    """
    plots absolute EMG values for max, mean and median frequency
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    x = range(len(labels))
    width = 0.35

    # plot metrics
    bars1 = ax.bar(
        [i - width / 2 for i in x], ohne_vals, width, label="ohne exo", color="#4F81BD"
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], mit_vals, width, label="mit exo", color="#C0504D"
    )

    # Annotate with percentage difference
    for i, (bar, ohne_val, mit_val) in enumerate(zip(bars2, ohne_vals, mit_vals)):
        if ohne_val != 0:
            perc_diff = 100 * (mit_val - ohne_val) / ohne_val
            sign = "+" if perc_diff >= 0 else ""
            annotation = f"{sign}{perc_diff:.1f}%"
        else:
            annotation = "n/a"
        ax.annotate(
            annotation,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=bar.get_facecolor(),
            fontweight="bold",
        )

    ax.set_ylabel("mV")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(ohne_vals + mit_vals) * 1.2)
    return bars1, bars2


def plot_emg_radar(muscles, base_path, out_path_prefix):
    """
    plots radar diagrams for all muscles and metrics max, mean and median frequency
    """
    metrics = [
        ("max", "Max (mV)"),
        ("mean", "Mean (mV)"),
        ("median frequency", "Median Freqency (Hz)"),
    ]
    muscle_labels = [abbr for _, _, abbr in muscles]  # Use abbreviations
    muscle_full_legend = [f"{abbr} - {name}" for _, name, abbr in muscles]

    # Prepare data for each exercise (0=aggregate, 1-6=exercises)
    for ex_idx in range(7):
        if ex_idx == 0:
            ex_name = "All Exercises"
            ex_file = os.path.join(base_path, "aggregate")
            out_path = f"{out_path_prefix}_radar_aggregate.png"
        else:
            ex_name = f"Exercise {ex_idx}"
            ex_file = os.path.join(base_path, f"Exercise {ex_idx}")
            out_path = f"{out_path_prefix}_radar_exercise{ex_idx}.png"

        # Collect values for all muscles
        ohne_vals = {metric: [] for metric, _ in metrics}
        mit_vals = {metric: [] for metric, _ in metrics}
        for muscle_file, _, _ in muscles:
            json_path = os.path.join(ex_file, muscle_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_total = data["all subjects"]
            ohne = all_total["ohne exo"]
            mit = all_total["mit exo"]
            for metric, _ in metrics:
                ohne_vals[metric].append(ohne[metric])
                mit_vals[metric].append(mit[metric])

        # Radar plot setup
        num_vars = len(muscle_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(18, 6))
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for idx, (metric, metric_label) in enumerate(metrics):
            ax = axs[idx]
            # Data for this metric
            values_ohne = ohne_vals[metric] + [ohne_vals[metric][0]]
            values_mit = mit_vals[metric] + [mit_vals[metric][0]]

            ax.plot(angles, values_ohne, label="ohne exo", color="#4F81BD", linewidth=2)
            ax.fill(angles, values_ohne, color="#4F81BD", alpha=0.15)
            ax.plot(angles, values_mit, label="mit exo", color="#C0504D", linewidth=2)
            ax.fill(angles, values_mit, color="#C0504D", alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([])

            # place lables a little outside the diagram, prevent overlapping
            label_radius = max(values_ohne + values_mit) * 1.22
            for angle, label in zip(angles[:-1], muscle_labels):
                ax.text(
                    angle,
                    label_radius,
                    label,
                    size=11,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )

            ax.set_title(metric_label, size=14, y=1.1)
            ax.grid(True)

            # Annotate percentage change between no exo and with exo
            for i, angle in enumerate(angles[:-1]):

                if values_ohne[i] != 0:
                    perc = 100 * (values_mit[i] - values_ohne[i]) / values_ohne[i]
                    sign = "+" if perc >= 0 else ""
                    perc_text = f"{sign}{perc:.1f}%"
                else:
                    perc_text = ""
                higher = max(values_ohne[i], values_mit[i])
                offset = 0.07 * max(values_ohne + values_mit)
                ax.text(
                    angle,
                    higher + offset,
                    perc_text,
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_ylabel("", labelpad=20)

        # Add legend for abbreviations
        fig.text(
            1.02,
            0.5,
            "\n".join(muscle_full_legend),
            va="center",
            ha="left",
            fontsize=12,
            transform=fig.transFigure,
        )

        axs[0].legend(loc="upper left", bbox_to_anchor=(1.1, 1.1), fontsize=12)
        fig.suptitle(f"EMG Radar Diagram: {ex_name}", fontsize=18, y=1.08)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    """
    main execution: plot bar and radar diagrams for all muscles and exercises
    """

    muscles = [
        ("Biceps_femoris_left.json", "Biceps femoris left", "b.f.l."),
        ("Biceps_femoris_right.json", "Biceps femoris right", "b.f.r."),
        ("Quadriceps_left.json", "Quadriceps left", "q.l."),
        ("Quadriceps_right.json", "Quadriceps right", "q.r."),
        ("Gluteus_maximus_left.json", "Gluteus maximus left", "g.m.l."),
        ("Gluteus_maximus_right.json", "Gluteus maximus right", "g.m.r."),
        ("Erector_spinae_left.json", "Erector spinae left", "e.s.l."),
        ("Erector_spinae_right.json", "Erector spinae right", "e.s.r."),
    ]

    base_path = "Data/EMG/evaluations"

    plot_emg_radar(muscles, base_path, os.path.join(base_path, "EMG"))

    for muscle_file, muscle_name, _ in muscles:
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
        bars1, bars2 = visualize_absolute_emg_values(ax_all, params[0][0], params[0][1])
        all_bars = (bars1, bars2)

        # Next 6: Exercises 1-6 in 2 rows, 3 columns each
        axes = []
        for i in range(6):
            row = 1 + i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            visualize_absolute_emg_values(ax, params[i + 1][0], params[i + 1][1])
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
            os.path.join(base_path, f"EMG_bars_{muscle_name.replace(' ', '_')}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
