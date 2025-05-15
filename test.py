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

l_v = np.array([1, 5, 6, 2, 0, 9, 14])
l_i = np.array([0, 1, 2])

# Create a mask of True for all indices
mask = np.ones(len(l_v), dtype=bool)

# Set False at positions to remove
mask[l_i] = False

# Apply mask to original array
l_o = l_v[mask]

print(l_o)
