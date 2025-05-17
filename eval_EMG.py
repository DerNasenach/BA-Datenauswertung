import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

SAMPLING_RATE = 2148.1481  # Hz


def compute_median_frequency(signal):
    # Compute power Spectral Density using welch
    freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=1024)

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

    return median_freq


def compute_rectified_max_mean(signal):
    signal_rectified = np.abs(signal)
    return np.max(signal_rectified), np.mean(signal_rectified)


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
            file_prefix_path_with_exo + subject_number_str + ".csv",
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

            values_no_exo_rms = root_mean_square(values_no_exo_cleaned)
            values_with_exo_rms = root_mean_square(values_with_exo_cleaned)

            no_exo_max = np.max(values_no_exo_rms)
            no_exo_mean = np.mean(values_no_exo_rms)
            no_exo_median_freq = compute_median_frequency(values_no_exo_cleaned)

            with_exo_max = np.max(values_with_exo_rms)
            with_exo_mean = np.mean(values_with_exo_rms)
            with_exo_median_freq = compute_median_frequency(values_with_exo_cleaned)

            change_max = 1 - with_exo_max / no_exo_max
            change_mean = 1 - with_exo_mean / no_exo_mean
            change_median_freq = 1 - with_exo_median_freq / no_exo_median_freq

            evaluation_exercise.append((change_max, change_mean, change_median_freq))

        evaluation_subject.append(evaluation_exercise)

    return evaluation_subject


def compare_statistics(evals):
    muscles_stats = []
    for muscle_number in range(0, 8):
        muscle_max_changes = []
        muscle_mean_changes = []
        muscle_median_freq_changes = []
        for subject_number in range(len(evals)):
            for exercise_number in range(len(evals[subject_number])):
                muscle_max_changes.append(
                    evals[subject_number][exercise_number][muscle_number][0]
                )
                muscle_mean_changes.append(
                    evals[subject_number][exercise_number][muscle_number][1]
                )
                muscle_median_freq_changes.append(
                    evals[subject_number][exercise_number][muscle_number][2]
                )
        print(muscle_max_changes)
        muscles_stats.append()
    return


subject_evals = []
for i in range(1, 3):
    subject_evals.append(eval_subject(i))
# print(subject_evals)

compare_statistics(subject_evals)


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
