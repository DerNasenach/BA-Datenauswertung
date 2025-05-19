import matplotlib.pyplot as plt
import json
import numpy as np

# The column the filtered signal in z-axis is stored
Z_FILTERED_COLUMN = 11
NUMBER_OF_SUBJECTS = 8


def extract_column(file_path, n=Z_FILTERED_COLUMN):
    data = []
    time = []
    with open(file_path, "r") as file:
        content = file.read().split("\n")
        name = content[16].split("\t")[n]

        for i in range(18, len(content) - 2, 2):
            line = content[i].split("\t")
            data.append(float(line[n]))
            time.append(float(line[0]))

    return name, data, time


def get_slices_of_subject(subject_number):
    subject_number_str = str(subject_number)
    folder_path = "Data/BKMP/Subject" + subject_number_str
    file_prefix_path = folder_path + "/subject" + subject_number_str + "_"
    json_path = file_prefix_path + "slices.json"
    # no_exo_path = file_prefix_path + "ohne_exo.acp"
    # with_exo_path = file_prefix_path + "mit_exo.acp"
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


def get_statistical_analysis(signal1, signal2):
    # TODO: implement analysis. Max, mean, powerspikes(?)

    return


def get_power_spikes(signal):
    # TODO: implement power spike detection
    return


def make_eval(signals):
    # print(signals)

    max_values_no_exo = []
    mean_values_no_exo = []
    power_spike_values_no_exo = []

    max_values_with_exo = []
    mean_values_with_exo = []
    power_spike_values_with_exo = []

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
        # print(round_no_exo)

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
        # power_spike_values_no_exo.append(round_no_exo_power_spikes)

        max_values_with_exo.append(round_with_exo_max)
        mean_values_with_exo.append(round_with_exo_mean)
        # power_spike_values_with_exo.append(round_with_exo_power_spikes)

    report_concat["concatenated evaluation"]["All Subjects total"] = {
        "ohne exo": {
            "mean of max": np.mean(max_values_no_exo),
            "mean of means": np.mean(mean_values_no_exo),
            # "mean of power spikes": np.mean(power_spike_values_no_exo),
        },
        "mit exo": {
            "mean of max": np.mean(max_values_with_exo),
            "mean of means": np.mean(mean_values_with_exo),
            # "mean of power spikes": np.mean(power_spike_values_with_exo),
        },
    }
    print(report_concat)

    report_concat["concatenated evaluation"]["statistics"] = {}
    # TODO: analysis to file

    # TODO: analysis over each exercise


slices = get_slices_all_subjects()
make_eval(slices)


"""
name, data, time = extract_column("./Data/BKMP/Subject8/subject8_mit_exo.acp", 11)
print(name)
plt.plot(time, data, label=name)
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Subject8 mit exo")

plt.grid(True)
plt.legend()

plt.show()
# plt.(figsize=(8, 4))
"""
