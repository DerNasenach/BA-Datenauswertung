import os
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.signal import welch
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import norm

# The column the filtered signal in z-axis is stored
Z_FILTERED_COLUMN = 11
NUMBER_OF_SUBJECTS = 8
HEADER_ROW = 16
DATA_START_ROW = 18


# extracts the n-th column of the specified file
def extract_column(file_path, n=Z_FILTERED_COLUMN, start_row=DATA_START_ROW):
    data = []
    time = []
    with open(file_path, "r") as file:
        content = file.read().split("\n")
        name = content[HEADER_ROW].split("\t")[n]

        for i in range(start_row, len(content) - 2, 2):
            line = content[i].split("\t")
            data.append(float(line[n]))
            time.append(float(line[0]))

    return name, data, time


# slices the acquired data of specified subject based on indices in the corresponding json
def get_slices_of_subject(subject_number):
    subject_number_str = str(subject_number)
    folder_path = "Data/BKMP/Subject" + subject_number_str
    file_prefix_path = folder_path + "/subject" + subject_number_str + "_"
    json_path = file_prefix_path + "slices.json"
    slices_file = open(json_path)
    slices_indices = json.load(slices_file)

    signal_subject_sliced = []

    for round_name, indices_round in slices_indices.items():
        acp_path = file_prefix_path + round_name + ".acp"

        _, signal_raw, _ = extract_column(acp_path)
        signal_round_sliced = []

        for _, indices_exercise in indices_round.items():
            signal_exercise_slice = []

            for i_start, i_end in indices_exercise:
                signal_exercise_slice += signal_raw[i_start:i_end]

            signal_round_sliced.append(np.array(signal_exercise_slice))

        signal_subject_sliced.append(signal_round_sliced)
    return signal_subject_sliced


def get_slices_all_subjects():
    slices = []
    for i in range(1, NUMBER_OF_SUBJECTS + 1):
        slices.append(get_slices_of_subject(i))
    return slices


# calculates basic statistical analysis over dataset, return report as dict
def get_statistical_analysis(differences, values1, values2):
    _, p = shapiro(differences)

    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # differences are normally distributed, use paired t-test
    if p > 0.05:
        t_stat, p = ttest_rel(values1, values2)

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
        stat, p = wilcoxon(values1, values2)

        # Compute effect size
        z = norm.ppf(1 - p / 2)
        r = z / np.sqrt(len(differences))

        effect_size = "Wilcoxon r-value : " + str(r)
        if r < 0.1:
            effect_size += ". no effect"
        elif r < 0.3:
            effect_size += ". small effect"
        elif r < 0.5:
            effect_size += ". medium effect"
        else:
            effect_size += ". large effect"

    ci_low = mean_diff - 1.96 * (std_diff / np.sqrt(len(differences)))
    ci_high = mean_diff + 1.96 * (std_diff / np.sqrt(len(differences)))

    return {
        "mean difference": mean_diff,
        "standart deviation": std_diff,
        "95'%' confidence interval low": ci_low,
        "95'%' confidence interval high": ci_high,
        "p-value": p,
        "effect size": effect_size,
    }


# calculates mean, max and statistical analysis differentiated over all exercises. Writes reports to json
def make_eval_over_exercises(signals):
    max_values_no_exo = [[] for _ in range(6)]
    mean_values_no_exo = [[] for _ in range(6)]

    max_values_with_exo = [[] for _ in range(6)]
    mean_values_with_exo = [[] for _ in range(6)]

    diffs_max_values = [[] for _ in range(6)]
    diffs_mean_values = [[] for _ in range(6)]

    reports = [
        {
            "Exercise "
            + str(n): {
                "Subject "
                + str(i): {
                    "ohne exo": {},
                    "mit exo": {},
                }
                for i in range(1, len(signals) + 1)
            }
        }
        for n in range(1, 7)
    ]

    for i, subject in enumerate(signals):
        for j, exercise in enumerate(subject[0]):
            max_value_no_exo = np.max(exercise)
            mean_value_no_exo = np.mean(exercise)

            reports[j][f"Exercise {j+1}"]["Subject " + str(i + 1)]["ohne exo"][
                "max"
            ] = max_value_no_exo
            reports[j]["Exercise " + str(j + 1)]["Subject " + str(i + 1)]["ohne exo"][
                "mean"
            ] = mean_value_no_exo

            max_values_no_exo[j].append(max_value_no_exo)
            mean_values_no_exo[j].append(mean_value_no_exo)

        for j, exercise in enumerate(subject[1]):
            max_value_with_exo = np.max(exercise)
            mean_value_with_exo = np.mean(exercise)

            reports[j]["Exercise " + str(j + 1)]["Subject " + str(i + 1)]["mit exo"][
                "max"
            ] = max_value_with_exo
            reports[j]["Exercise " + str(j + 1)]["Subject " + str(i + 1)]["mit exo"][
                "mean"
            ] = mean_value_with_exo

            max_values_with_exo[j].append(max_value_with_exo)
            mean_values_with_exo[j].append(mean_value_with_exo)

        # diffs_max_values.append(max_values_no_exo[i] - max_values_with_exo[i])
        # diffs_mean_values.append(mean_values_no_exo[i] - mean_values_with_exo[i])
    for ex in range(6):
        for subj in range(8):
            diffs_max_values[ex].append(
                max_values_no_exo[ex][subj] - max_values_with_exo[ex][subj]
            )
            diffs_mean_values[ex].append(
                mean_values_no_exo[ex][subj] - mean_values_with_exo[ex][subj]
            )

    # print(reports[1])
    for i in range(6):
        reports[i]["Exercise " + str(i + 1)]["All subjects"] = {
            "ohne exo": {
                "mean of max": np.mean(max_values_no_exo[i]),
                "mean of mean": np.mean(mean_values_no_exo[i]),
            },
            "mit exo": {
                "mean of max": np.mean(max_values_with_exo[i]),
                "mean of mean": np.mean(mean_values_with_exo[i]),
            },
        }
        reports[i]["Exercise " + str(i + 1)]["statistics"] = {
            "max": get_statistical_analysis(
                diffs_max_values[i], max_values_no_exo[i], max_values_with_exo[i]
            ),
            "mean": get_statistical_analysis(
                diffs_mean_values[i], mean_values_no_exo[i], mean_values_with_exo[i]
            ),
        }

        os.makedirs("Data/BKMP/evaluations", exist_ok=True)
        with open(
            f"Data/BKMP/evaluations/evaluations_exercise_{i+1}.json", "w"
        ) as file:
            json.dump(reports[i], file, indent=4)

    return


# calculates mean, max and statistical analysis concatenated over all exercises. Writes report to json
def make_eval_concat(signals):
    # print(signals)

    max_values_no_exo = []
    mean_values_no_exo = []
    power_spikes_values_no_exo = []

    max_values_with_exo = []
    mean_values_with_exo = []
    power_spikes_values_with_exo = []

    diffs_max_values = []
    diffs_mean_values = []
    diffs_power_spikes_values = []

    report_concat = {"concatenated evaluation": {}}
    # evaluate over both rounds concatenated
    for i, subject in enumerate(signals):
        round_no_exo = []
        round_with_exo = []

        for exercise in subject[0]:
            round_no_exo = np.concatenate((round_no_exo, exercise))
        for exercise in subject[1]:
            round_with_exo = np.concatenate((round_with_exo, exercise))

        round_no_exo_max = np.max(round_no_exo)
        round_no_exo_mean = np.mean(round_no_exo)
        # round_no_exo_power_spikes = get_power_spikes(round_no_exo)

        round_with_exo_max = np.max(round_with_exo)
        round_with_exo_mean = np.mean(round_with_exo)
        # round_with_exo_power_spikes = get_power_spikes(round_with_exo)

        diffs_max_values.append(round_no_exo_max - round_with_exo_max)
        diffs_mean_values.append(round_no_exo_mean - round_with_exo_mean)
        # diffs_power_spikes_values.append(round_no_exo_power_spikes - round_with_exo_power_spikes)

        report_concat["concatenated evaluation"]["Subject" + str(i)] = {
            "ohne exo": {
                "max": round_no_exo_max,
                "mean": round_no_exo_mean,
                # "power spikes": round_no_exo_power_spikes,
            },
            "mit exo": {
                "max": round_with_exo_max,
                "mean": round_with_exo_mean,
                # "power spikes": round_with_exo_power_spikes,
            },
        }
        max_values_no_exo.append(round_no_exo_max)
        mean_values_no_exo.append(round_no_exo_mean)
        # power_spikes_values_no_exo.append(round_no_exo_power_spikes)

        max_values_with_exo.append(round_with_exo_max)
        mean_values_with_exo.append(round_with_exo_mean)
        # power_spikes_values_with_exo.append(round_with_exo_power_spikes)

    report_concat["concatenated evaluation"]["All Subjects total"] = {
        "ohne exo": {
            "mean of max": np.mean(max_values_no_exo),
            "mean of means": np.mean(mean_values_no_exo),
            # "mean of power spikes": np.mean(power_spikes_values_no_exo),
        },
        "mit exo": {
            "mean of max": np.mean(max_values_with_exo),
            "mean of means": np.mean(mean_values_with_exo),
            # "mean of power spikes": np.mean(power_spikes_values_with_exo),
        },
    }
    report_concat["concatenated evaluation"]["statistics"] = {}

    report_concat["concatenated evaluation"]["statistics"]["max"] = (
        get_statistical_analysis(
            diffs_max_values, max_values_no_exo, max_values_with_exo
        )
    )
    report_concat["concatenated evaluation"]["statistics"]["mean"] = (
        get_statistical_analysis(
            diffs_mean_values, mean_values_no_exo, mean_values_with_exo
        )
    )
    """
    report_concat["concatenated evaluation"]["statistics"]["power spikes"] = (
        get_statistical_analysis(
            diffs_power_spikes_values, power_spikes_values_no_exo, power_spikes_values_with_exo
        )
    )"""
    print(report_concat)

    os.makedirs("Data/BKMP/evaluations", exist_ok=True)
    with open("Data/BKMP/evaluations/evaluations_concat.json", "w") as file:
        json.dump(report_concat, file, indent=4)


slices = get_slices_all_subjects()
make_eval_concat(slices)
make_eval_over_exercises(slices)

"""
name, data, time = extract_column("./Data/BKMP/Subject5/subject5_ohne_exo.acp", 11)
print(name)
plt.plot(time, data, label=name)
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Subject5 ohne exo")

plt.grid(True)
plt.legend()

plt.show()
# plt.(figsize=(8, 4))
"""
