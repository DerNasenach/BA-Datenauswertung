import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.signal import welch
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import norm
from scipy.ndimage import generic_filter

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

SAMPLING_RATE = 2148.1481  # Hz
NUMBER_OF_SUBJECTS = 8
NUMBER_OF_EXERCISES = 6
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
VERBOSE = True
MULTI_PROCESSING = True


def get_median_frequency(signal):
    # Compute power Spectral Density using welch
    freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=1024)

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

    return median_freq


# returns a list of int with indices of detected anomalies in the input list
def detect_anomalies_iforest(
    time_series, window_size=8000, stride=800, std_factor=4.5, plot=False
):
    if VERBOSE:
        print("Starting anomaly detection")
    ts = np.array(time_series).reshape(-1, 1)
    n = len(ts)
    score_sum = np.zeros(n)
    score_count = np.zeros(n)

    model = IsolationForest(contamination="auto", random_state=42)

    # Slide windows
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        # print(f"detecting for window {start} - {end}")
        window_data = ts[start:end]

        model.fit(window_data)
        window_scores = model.decision_function(window_data)  # shape: (window_size,)

        # Aggregate scores per index
        score_sum[start:end] += window_scores
        score_count[start:end] += 1

    # Avoid division by zero
    valid = score_count > 0
    averaged_scores = np.full(n, np.nan)
    averaged_scores[valid] = score_sum[valid] / score_count[valid]

    # Compute global threshold from non-NaN averaged scores
    score_mean = np.nanmean(averaged_scores)
    score_std = np.nanstd(averaged_scores)
    threshold = score_mean - std_factor * score_std

    # Label anomalies
    anomaly_mask = averaged_scores < threshold
    anomaly_indices = np.where(anomaly_mask)[0]
    if VERBOSE:
        print(f"End of anomaly detection. {len(anomaly_indices)} anomalies found")

    # Plot
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(ts, label="Time Series")
        plt.scatter(
            anomaly_indices, ts[anomaly_indices], color="red", label="Anomalies"
        )
        plt.title("Adaptive Isolation Forest (Overlapping Windows)")
        plt.legend()
        plt.show()

    return anomaly_indices


# removes anomalies from time series, returns altered time series
def remove_anomalies(time_series: np.array):
    if np.max(np.abs(time_series)) < 1:
        return time_series
    anomalies = detect_anomalies_iforest(time_series)

    mask = np.ones(len(time_series), dtype=bool)
    mask[anomalies] = False

    return time_series[mask]


# returns the Root-Mean-Square of a time series
def rms(time_series):
    return np.sqrt(np.mean(time_series**2))


# returns a windowed Root-Mean-Square of a time series
def wrms(time_series, window_size=50):
    return generic_filter(time_series, rms, size=window_size)


# reads and formats the sliced data of specified subject, computes metrics
def get_metrics_subject(subject_number: int):
    if VERBOSE:
        print(f"evaluating Subject {subject_number}")
    subject_number_str = str(subject_number)
    folder_path = "Data/EMG/Subject" + subject_number_str + "/sliced/"
    file_prefix_path_no_exo = folder_path + "ohne_exo_exercise"
    file_prefix_path_with_exo = folder_path + "mit_exo_exercise"

    evaluation_subject = []

    for i in range(1, 7):
        if VERBOSE:
            print(f"reading exercise {i}")
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
            if VERBOSE:
                print(f"reading muscle {MUSCLE_NAMES[j]}")
            muscle_name = str(header[j])
            values_no_exo_cleaned = remove_anomalies(matrix_no_exo[j])
            values_with_exo_cleaned = remove_anomalies(matrix_with_exo[j])

            # TODO: evaluate wether abs or windowed rms should be used
            values_no_exo_wrms = wrms(values_no_exo_cleaned)
            values_with_exo_wrms = wrms(values_with_exo_cleaned)

            no_exo_max = np.max(values_no_exo_wrms)
            no_exo_mean = np.mean(values_no_exo_wrms)
            no_exo_median_freq = get_median_frequency(values_no_exo_cleaned)

            with_exo_max = np.max(values_with_exo_wrms)
            with_exo_mean = np.mean(values_with_exo_wrms)
            with_exo_median_freq = get_median_frequency(values_with_exo_cleaned)

            evaluation_exercise.append(
                (
                    (no_exo_max, with_exo_max),
                    (no_exo_mean, with_exo_mean),
                    (no_exo_median_freq, with_exo_median_freq),
                    (values_no_exo_wrms, values_with_exo_wrms),
                )
            )

        evaluation_subject.append(evaluation_exercise)

    return evaluation_subject


# takes two lists of max, mean and median freq values
def get_analysis(muscle_number: int, values, differences):
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
    return report


def make_statistics_analysis_from_metrics(evals):
    if VERBOSE:
        print("generating statistical analysis")
    muscles_values = []
    muscles_differences = []
    reports = [
        {
            f"Exercise {n}": {
                MUSCLE_NAMES[i]: {
                    f"Subject {j}": {
                        "ohne exo": {},
                        "mit exo": {},
                    }
                    for j in range(1, len(evals) + 1)
                }
                for i in range(0, 8)
            }
        }
        for n in range(1, NUMBER_OF_EXERCISES + 1)
    ]
    # Get max, mean and median freq for each muscle over all exercises
    if VERBOSE:
        print("  evaluating over each exercise over each muscle")
    for muscle_number in range(0, 8):
        if VERBOSE:
            print(f"    evaluating muscle {MUSCLE_NAMES[muscle_number]}")
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

                reports[exercise_number][f"Exercise {exercise_number + 1}"][
                    MUSCLE_NAMES[muscle_number]
                ][f"Subject {subject_number +1}"]["ohne exo"] = {
                    "max": evals[subject_number][exercise_number][muscle_number][0][0],
                    "mean": evals[subject_number][exercise_number][muscle_number][1][0],
                    "median frequency": evals[subject_number][exercise_number][
                        muscle_number
                    ][2][0],
                }
                reports[exercise_number][f"Exercise {exercise_number + 1}"][
                    MUSCLE_NAMES[muscle_number]
                ][f"Subject {subject_number +1}"]["mit exo"] = {
                    "max": evals[subject_number][exercise_number][muscle_number][0][1],
                    "mean": evals[subject_number][exercise_number][muscle_number][1][1],
                    "median frequency": evals[subject_number][exercise_number][
                        muscle_number
                    ][2][1],
                }
                reports[exercise_number][f"Exercise {exercise_number + 1}"][
                    MUSCLE_NAMES[muscle_number]
                ][f"Subject {subject_number +1}"]["difference"] = {
                    "reduction max in %": (
                        1
                        - (
                            evals[subject_number][exercise_number][muscle_number][0][1]
                            / evals[subject_number][exercise_number][muscle_number][0][
                                0
                            ]
                        )
                    )
                    * 100,
                    "reduction mean in %": (
                        1
                        - (
                            evals[subject_number][exercise_number][muscle_number][1][1]
                            / evals[subject_number][exercise_number][muscle_number][1][
                                0
                            ]
                        )
                    )
                    * 100,
                    "reduction median frequency in %": (
                        1
                        - (
                            evals[subject_number][exercise_number][muscle_number][2][1]
                            / evals[subject_number][exercise_number][muscle_number][2][
                                0
                            ]
                        )
                    )
                    * 100,
                }

        # print(muscle_max_values)
        muscles_values.append(
            (muscle_max_values, muscle_mean_values, muscle_median_freq_values)
        )
        muscles_differences.append(
            (muscle_max_diffs, muscle_mean_diffs, muscle_median_freq_diffs)
        )
        if VERBOSE:
            print("    evaluating over all subjects")
        for exercise_dict in reports:
            for _, d in exercise_dict.items():
                for muscle_name, muscle_values in d.items():
                    if muscle_name != MUSCLE_NAMES[muscle_number]:
                        continue
                    max_total_no_exo = []
                    mean_total_no_exo = []
                    median_freq_total_no_exo = []

                    max_total_with_exo = []
                    mean_total_with_exo = []
                    median_freq_total_with_exo = []
                    for _, subject in muscle_values.items():
                        max_total_no_exo.append(subject["ohne exo"]["max"])
                        mean_total_no_exo.append(subject["ohne exo"]["mean"])
                        median_freq_total_no_exo.append(
                            subject["ohne exo"]["median frequency"]
                        )

                        max_total_with_exo.append(subject["mit exo"]["max"])
                        mean_total_with_exo.append(subject["mit exo"]["mean"])
                        median_freq_total_with_exo.append(
                            subject["mit exo"]["median frequency"]
                        )
                    muscle_values["All Subjects"] = {
                        "ohne exo": {
                            "mean of max": np.mean(max_total_no_exo),
                            "mean of means": np.mean(mean_total_no_exo),
                            "mean of meadian frequency": np.mean(
                                median_freq_total_no_exo
                            ),
                        },
                        "mit exo": {
                            "mean of max": np.mean(max_total_with_exo),
                            "mean of means": np.mean(mean_total_with_exo),
                            "mean of meadian frequency": np.mean(
                                median_freq_total_with_exo
                            ),
                        },
                        "difference": {
                            "reduction mean of max in %": (
                                1
                                - (
                                    np.mean(max_total_with_exo)
                                    / np.mean(max_total_no_exo)
                                )
                            )
                            * 100,
                            "reduction mean of means in %": (
                                1
                                - (
                                    np.mean(mean_total_with_exo)
                                    / np.mean(mean_total_no_exo)
                                )
                            )
                            * 100,
                            "reduction mean of median frequency in %": (
                                1
                                - (
                                    np.mean(median_freq_total_with_exo)
                                    / np.mean(median_freq_total_no_exo)
                                )
                            )
                            * 100,
                        },
                    }

    for i, exercise_dict in enumerate(reports):
        for exercise, d in exercise_dict.items():
            os.makedirs(f"Data/EMG/evaluations/{exercise}", exist_ok=True)
            for muscle, data in d.items():

                with open(
                    f"Data/EMG/evaluations/{exercise}/{muscle}.json", "w"
                ) as file:
                    json.dump(data, file, indent=4)

    # for i in range(0, 8):
    #   make_statistical_analysis(i, muscles_values[i], muscles_differences[i])

    # Aggregate evaluation
    if VERBOSE:
        print("evaluating over aggregated exercises")
    aggregate_report = {MUSCLE_NAMES[i]: {} for i in range(8)}

    for muscle_number in range(8):
        if VERBOSE:
            print(f"    evaluating muscle {MUSCLE_NAMES[muscle_number]}")
        for subject_number in range(len(evals)):
            all_raw_no_exo = []
            all_raw_with_exo = []

            for exercise_number in range(len(evals[subject_number])):
                all_raw_no_exo.extend(
                    evals[subject_number][exercise_number][muscle_number][3][0]
                )
                all_raw_with_exo.extend(
                    evals[subject_number][exercise_number][muscle_number][3][1]
                )

            all_raw_no_exo = np.array(all_raw_no_exo)
            all_raw_with_exo = np.array(all_raw_with_exo)

            aggregate_report[MUSCLE_NAMES[muscle_number]][
                f"Subject {subject_number + 1}"
            ] = {
                "ohne exo": {
                    "max": np.max(all_raw_no_exo),
                    "mean": np.mean(all_raw_no_exo),
                    "median frequency": np.median(all_raw_no_exo),
                },
                "mit exo": {
                    "max": np.max(all_raw_with_exo),
                    "mean": np.mean(all_raw_with_exo),
                    "median frequency": np.median(all_raw_with_exo),
                },
                "difference": {
                    "reduction max in %": (
                        1 - np.max(all_raw_with_exo) / np.max(all_raw_no_exo)
                    )
                    * 100,
                    "reduction mean in %": (
                        1 - np.mean(all_raw_with_exo) / np.mean(all_raw_no_exo)
                    )
                    * 100,
                    "reduction median frequency in %": (
                        1 - np.median(all_raw_with_exo) / np.median(all_raw_no_exo)
                    )
                    * 100,
                },
            }

        # Aggregate across all subjects
        all_subjects_raw_no_exo = []
        all_subjects_raw_with_exo = []

        for subject_number in range(len(evals)):
            for exercise_number in range(len(evals[subject_number])):
                all_subjects_raw_no_exo.extend(
                    evals[subject_number][exercise_number][muscle_number][3][0]
                )
                all_subjects_raw_with_exo.extend(
                    evals[subject_number][exercise_number][muscle_number][3][1]
                )

        all_subjects_raw_no_exo = np.array(all_subjects_raw_no_exo)
        all_subjects_raw_with_exo = np.array(all_subjects_raw_with_exo)

        aggregate_report[MUSCLE_NAMES[muscle_number]]["All Subjects"] = {
            "ohne exo": {
                "max": float(np.max(all_subjects_raw_no_exo)),
                "mean": float(np.mean(all_subjects_raw_no_exo)),
                "median frequency": float(np.median(all_subjects_raw_no_exo)),
            },
            "mit exo": {
                "max": float(np.max(all_subjects_raw_with_exo)),
                "mean": float(np.mean(all_subjects_raw_with_exo)),
                "median frequency": float(np.median(all_subjects_raw_with_exo)),
            },
            "difference": {
                "reduction max in %": (
                    1
                    - np.max(all_subjects_raw_with_exo)
                    / np.max(all_subjects_raw_no_exo)
                )
                * 100,
                "reduction mean in %": (
                    1
                    - np.mean(all_subjects_raw_with_exo)
                    / np.mean(all_subjects_raw_no_exo)
                )
                * 100,
                "reduction median frequency in %": (
                    1
                    - np.median(all_subjects_raw_with_exo)
                    / np.median(all_subjects_raw_no_exo)
                )
                * 100,
            },
        }

    # Save aggregate report
    os.makedirs("Data/EMG/evaluations/aggregate", exist_ok=True)
    for muscle, data in aggregate_report.items():
        with open(f"Data/EMG/evaluations/aggregate/{muscle}.json", "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    if MULTI_PROCESSING:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            subject_evals = pool.map(get_metrics_subject, range(1, 9))
    else:
        subject_evals = []
        for i in range(1, 9):
            subject_evals.append(get_metrics_subject(i))
    make_statistics_analysis_from_metrics(subject_evals)
