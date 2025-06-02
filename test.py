import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


"""
with open("test1.csv", mode="r", newline="", encoding="utf-8") as infile:
    csv_reader = csv.reader(infile)
    rows = list(csv_reader)
    print(rows[10])
    print(rows[10][0])
    print(float(rows[10][0].replace(" ", "").split(";")[0]))
    print()"""


def parse_file(filename):
    with open(filename, "r") as file:
        data = file.read().strip().split("\n\n")

    result = []

    for section in data:
        lines = section.splitlines()
        header = lines[0]
        pairs = []

        for line in lines[1:]:
            numbers = tuple(map(int, line.split(", ")))
            pairs.append(numbers)

        result.append((header, pairs))

    return result


def compute_median_frequency(signal, sampling_rate):
    # Compute power Spectral Density using welch
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=1024)

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

    return median_freq, freqs, psd


"""
with open(
    "./Data/Subject2/EMG/sliced/mit_exo_exercises_combined.csv",
    mode="r",
    newline="",
    encoding="utf-8",
) as infile:
    csv_reader = csv.reader(infile)
    rows = list(csv_reader)

sampling_rate = 2148.1481  # Hz
signal = []
for row in rows[4:]:
    signal.append(float(row[0].replace(" ", "").split(";")[6]))
median_freq, freqs, psd = compute_median_frequency(signal, sampling_rate)

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(time_series, contamination=0.01, plot=False):
    """
    Detect anomalies in a 1D time series using Isolation Forest.

    Parameters:
        time_series (array-like): 1D array of numerical values.
        contamination (float): Proportion of anomalies in the data.
        plot (bool): If True, plot the time series with anomalies highlighted.

    Returns:
        dict: Contains anomaly scores, labels, and indices of detected anomalies.
    """
    # Ensure input is numpy array and reshape for sklearn (n_samples, n_features)
    time_series = np.array(time_series).reshape(-1, 1)

    # Fit Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(time_series)

    # Predict anomalies
    labels = model.predict(time_series)  # -1 for anomaly, 1 for normal
    scores = model.decision_function(time_series)  # The lower, the more anomalous

    # Collect anomaly indices
    anomaly_indices = np.where(labels == -1)[0]

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(time_series, label="Time Series")
        plt.scatter(
            anomaly_indices,
            time_series[anomaly_indices],
            color="red",
            label="Anomalies",
        )
        plt.title("Anomaly Detection using Isolation Forest")
        plt.legend()
        plt.show()

    return {"labels": labels, "scores": scores, "anomaly_indices": anomaly_indices}


"""
# Example usage:
if __name__ == "__main__":
    # Generate synthetic data with anomalies
    np.random.seed(0)
    ts = np.sin(np.linspace(0, 20, 500)) + np.random.normal(0, 0.1, 500)
    ts[100] += 3  # Inject anomaly
    ts[101] += 3  # Inject anomaly
    ts[102] += 3  # Inject anomaly
    ts[103] += 3  # Inject anomaly
    ts[400] -= 3  # Inject another anomaly

    result = detect_anomalies_isolation_forest(ts, contamination=0.01, plot=True)
    print("Anomalies detected at indices:", result["anomaly_indices"])
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def detect_anomalies_iforest_manual_threshold(
    time_series, threshold=None, std_factor=2.0, plot=False
):
    """
    Detect anomalies in a 1D time series using Isolation Forest with manual score thresholding.

    Parameters:
        time_series (array-like): 1D array of numerical values.
        threshold (float or None): Manual threshold on the anomaly score. If None, use mean - std_factor * std.
        std_factor (float): Number of standard deviations below the mean for dynamic thresholding.
        plot (bool): Whether to plot the time series with anomalies.

    Returns:
        dict: Contains anomaly indices, scores, and anomaly mask.
    """
    time_series = np.array(time_series).reshape(-1, 1)

    # Fit Isolation Forest without contamination (uses default 0.1 but we override result)
    model = IsolationForest(contamination="auto", random_state=42)
    model.fit(time_series)

    # Get anomaly scores (higher = more normal, lower = more abnormal)
    scores = model.decision_function(time_series)

    # Dynamic threshold if not provided
    if threshold is None:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score - std_factor * std_score

    # Points with score below threshold are anomalies
    anomaly_mask = scores < threshold
    anomaly_indices = np.where(anomaly_mask)[0]

    # Plotting
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(time_series, label="Time Series")
        plt.scatter(
            anomaly_indices,
            time_series[anomaly_indices],
            color="red",
            label="Anomalies",
        )
        plt.title("Isolation Forest Anomaly Detection (Manual Threshold)")
        plt.legend()
        plt.show()

    return {
        "scores": scores,
        "threshold": threshold,
        "anomaly_mask": anomaly_mask,
        "anomaly_indices": anomaly_indices,
    }


# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    ts = np.random.normal(0, 1, 500)
    ts[50] += 6
    ts[300] -= 7
    ts[450] += 8

    result = detect_anomalies_iforest_manual_threshold(ts, std_factor=3, plot=True)
    print("Anomalies at indices:", result["anomaly_indices"])
