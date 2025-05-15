import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

SAMPLING_RATE = 2148.1481  # Hz


def compute_median_frequency(signal, sampling_rate):
    # Compute power Spectral Density using welch
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=1024)

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

    return median_freq, freqs, psd


def compute_rectified_max_mean(signal):
    signal_rectified = np.abs(signal)
    return np.max(signal_rectified), np.mean(signal_rectified)


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


make_eval("./Data/EMG/Subject1/sliced", ".csv")
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
