import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import norm

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

SAMPLING_RATE = 2148.1481  # Hz

MUSCLE_NAMES = {
    0: "Quadriceps left",
    1: "Quadriceps right",
    2: "Biceps femoris right",
    3: "Gluteus maximus right",
    4: "Biceps femoris left",
    5: "Erector spinae left",
    6: "Gluteus maximus left",
    7: "Erector spinae right",
}


def compute_median_frequency(signal):
    # Compute power Spectral Density using welch
    freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=1024)

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

    return median_freq


"""
def make_eval(path, ext):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    # print(files)

    muscles = np.genfromtxt(
        path + "/" + files[0], delimiter=";", dtype=str, max_rows=1
    )[
        ::2
    ]  # TODO: sanitize
    muscles = np.insert(muscles, [0], [" ", " "])
    # print(muscles)

    out_table = muscles.copy()
    for file in files:
        print(file)

        data = np.genfromtxt(
            path + "/" + file, delimiter=";", dtype=float, skip_header=4
        ).T.copy()[::2]
        # print(str(data))

        max_values = np.array([file, "max mV rect"])
        mean_values = np.array([" ", "mean mV rect"])
        median_frequencies = np.array([" ", "median frequency"])
        for muscle in data:
            max, mean = compute_rectified_max_mean(muscle)
            max_values = np.append(max_values, str(max))
            mean_values = np.append(mean_values, str(mean))
            median_frequency, _, _ = compute_median_frequency(muscle, SAMPLING_RATE)
            median_frequencies = np.append(median_frequencies, str(median_frequency))
        # print(max_values)
        # print(mean_values)
        out_table = np.vstack((out_table, max_values, mean_values, median_frequencies))
        # print(out_table)

    # print(out_table)
    np.savetxt(
        path + "/evaluation" + ext,
        out_table,
        delimiter=";",
        fmt="%s",
    )
    return
"""


# returns a list of int with indices of detected anomalies in the input list
def get_anomalies(time_series: np.array):
    # TODO: implement
    return np.array([1])


# removes anomalies from time series, returns altered time series
def remove_anomalies(time_series: np.array):
    anomalies = get_anomalies(time_series)

    mask = np.ones(len(time_series), dtype=bool)
    mask[anomalies] = False

    return time_series[mask]


# returns the RMS of a time series
def root_mean_square(time_series):
    return time_series


# def eval_exercise()


def eval_subject(subject_number: int):
    subject_number_str = str(subject_number)
    folder_path = "Data/EMG/Subject" + subject_number_str + "/sliced/"
    file_prefix_path_no_exo = folder_path + "ohne_exo_exercise"
    file_prefix_path_with_exo = folder_path + "mit_exo_exercise"

    evaluation_subject = []

    for i in range(1, 7):
        with open(
            file_prefix_path_no_exo + str(i) + ".csv",
            mode="r",
            newline="",
            encoding="utf-8",
        ) as infile_no_exo:
            csv_reader_no_exo = csv.reader(infile_no_exo, delimiter=",")
            list_no_exo = list(csv_reader_no_exo)
            header = list_no_exo[0]
            matrix_no_exo = np.transpose(np.array(list_no_exo[1:], dtype=float))

        with open(
            file_prefix_path_with_exo + str(i) + ".csv",
            mode="r",
            newline="",
            encoding="utf-8",
        ) as infile_with_exo:
            csv_reader_with_exo = csv.reader(infile_with_exo, delimiter=",")
            matrix_with_exo = np.transpose(
                np.array(list(csv_reader_with_exo)[1:], dtype=float)
            )

        evaluation_exercise = []

        for j in range(0, 8):
            muscle_name = str(header[j])
            values_no_exo_cleaned = remove_anomalies(matrix_no_exo[j])
            values_with_exo_cleaned = remove_anomalies(matrix_with_exo[j])

            # TODO: evaluate wether abs or windowed rms should be used
            # values_no_exo_rms = root_mean_square(values_no_exo_cleaned)
            # values_with_exo_rms = root_mean_square(values_with_exo_cleaned)
            values_no_exo_abs = np.abs(values_no_exo_cleaned)
            values_with_exo_abs = np.abs(values_with_exo_cleaned)

            no_exo_max = np.max(values_no_exo_abs)
            no_exo_mean = np.mean(values_no_exo_abs)
            no_exo_median_freq = compute_median_frequency(values_no_exo_cleaned)

            with_exo_max = np.max(values_with_exo_abs)
            with_exo_mean = np.mean(values_with_exo_abs)
            with_exo_median_freq = compute_median_frequency(values_with_exo_cleaned)

            # change_max = 1 - with_exo_max / no_exo_max
            # change_mean = 1 - with_exo_mean / no_exo_mean
            # change_median_freq = 1 - with_exo_median_freq / no_exo_median_freq

            evaluation_exercise.append(
                (
                    (no_exo_max, with_exo_max),
                    (no_exo_mean, with_exo_mean),
                    (no_exo_median_freq, with_exo_median_freq),
                )
            )

        evaluation_subject.append(evaluation_exercise)

    return evaluation_subject


def make_statistical_analysis(muscle_number: int, values, differences):
    VALUE_STRINGS = {0: "max", 1: "mean", 2: "median frequency"}
    report = {MUSCLE_NAMES[muscle_number]: {}}

    for i in range(len(differences)):
        # Use shapiro-wilk test to test for normality
        stat, p = shapiro(differences[i])

        mean_diff = np.mean(differences[i])
        std_diff = np.std(differences[i], ddof=1)

        # differences are normally distributed, use paired t-test
        if p > 0.05:
            t_stat, p = ttest_rel(
                [a for (a, b) in values[i]], [b for (a, b) in values[i]]
            )

            # Compute Cohen's d-value for effect size
            d = mean_diff / std_diff
            effect_size = "Cohen d-value : " + str(d)
            if d < 0.2:
                effect_size += ". no effect"
            elif d < 0.5:
                effect_size += ". small effect"
            elif d < 0.8:
                effect_size += ". medium effect"
            else:
                effect_size += ". large effect"

        # differences are not normally distributed, use wilcoxon signed-rank test
        else:
            stat, p = wilcoxon([a for (a, b) in values[i]], [b for (a, b) in values[i]])

            # Compute effect size
            z = norm.ppf(1 - p / 2)
            r = z / np.sqrt(len(differences[i]))

            effect_size = "Wilcoxon r-value : " + str(r)
            if r < 0.1:
                effect_size += ". no effect"
            elif r < 0.3:
                effect_size += ". small effect"
            elif r < 0.5:
                effect_size += ". medium effect"
            else:
                effect_size += ". large effect"

        ci_low = mean_diff - 1.96 * (std_diff / np.sqrt(len(differences[i])))
        ci_high = mean_diff + 1.96 * (std_diff / np.sqrt(len(differences[i])))

        report[MUSCLE_NAMES[muscle_number]][VALUE_STRINGS[i]] = {
            "mean difference": mean_diff,
            "standart deviation": std_diff,
            "95'%' confidence interval low": ci_low,
            "95'%' confidence interval high": ci_high,
            "p-value": p,
            "effect size": effect_size,
        }
        # print("muscle: " + str(muscle_number))
        # print("p: " + str(p_value))
        # print("stat: " + str(stat))
    os.makedirs("Data/EMG/evaluations/per_muscle", exist_ok=True)
    with open(
        "Data/EMG/evaluations/per_muscle/" + MUSCLE_NAMES[muscle_number] + ".json", "w"
    ) as file:
        json.dump(report, file, indent=4)
    return


def compare_muscles_over_all_exercises(evals):
    muscles_values = []
    muscles_differences = []

    # Get max, mean and median freq for each muscle over all exercises
    for muscle_number in range(0, 8):
        muscle_max_values = []
        muscle_mean_values = []
        muscle_median_freq_values = []

        muscle_max_diffs = []
        muscle_mean_diffs = []
        muscle_median_freq_diffs = []
        for subject_number in range(len(evals)):
            for exercise_number in range(len(evals[subject_number])):
                muscle_max_values.append(
                    evals[subject_number][exercise_number][muscle_number][0]
                )
                muscle_mean_values.append(
                    evals[subject_number][exercise_number][muscle_number][1]
                )
                muscle_median_freq_values.append(
                    evals[subject_number][exercise_number][muscle_number][2]
                )

                muscle_max_diffs.append(
                    evals[subject_number][exercise_number][muscle_number][0][0]
                    - evals[subject_number][exercise_number][muscle_number][0][1]
                )
                muscle_mean_diffs.append(
                    evals[subject_number][exercise_number][muscle_number][1][0]
                    - evals[subject_number][exercise_number][muscle_number][1][1]
                )
                muscle_median_freq_diffs.append(
                    evals[subject_number][exercise_number][muscle_number][2][0]
                    - evals[subject_number][exercise_number][muscle_number][2][1]
                )

        # print(muscle_max_values)
        muscles_values.append(
            (muscle_max_values, muscle_mean_values, muscle_median_freq_values)
        )
        muscles_differences.append(
            (muscle_max_diffs, muscle_mean_diffs, muscle_median_freq_diffs)
        )
    # print(muscles_values)
    for i in range(0, 8):
        make_statistical_analysis(i, muscles_values[i], muscles_differences[i])

    return


subject_evals = []
for i in range(1, 9):
    subject_evals.append(eval_subject(i))
# print(subject_evals)

compare_muscles_over_all_exercises(subject_evals)


# make_eval("./Data/EMG/Subject1/sliced", ".csv")
# make_analysis("./Data/Subject2/EMG/sliced/mit_exo_exercises_combined.csv")
"""
print(f"Median Frequency: {median_freq:.2f} Hz")
# Plot PSD
plt.figure(figsize=(8, 4))
plt.semilogy(freqs, psd)
plt.axvline(
    median_freq,
    color="r",
    linestyle="--",
    label=f"Median Frequency = {median_freq:.2f} Hz",
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Power Spectral Density and Median Frequency")
plt.legend()
plt.grid()
plt.show()
"""
