import csv
import os
import json
import numpy as np

# slices the .csv data in the folders Data/EMG/Subject{n} per slice indices in Data/EMG/Subject{n}/subject{n}_slices.json

ROW_HEADER_START = 3
ROW_HEADER_END = 4
ROW_DATA_START = 7


def replace_comma(input_path):
    with open(input_path, "r") as file:
        content = file.read()
    content = content.replace(",", ".")

    with open(input_path, "w") as file:
        file.write(content)

    print(f"commas have been replaced with periods in {input_path}")


# returns a list of int with indices of detected anomalies in the input list
def get_anomalies(time_series: np.array):
    return time_series


# removes anomalies from every n*2 column in the input matrix
def remove_anomalies(matrix: np.array):
    matrix = np.transpose(matrix)

    for i in range(0, 8):
        anomalies = get_anomalies(matrix[2 * i])

        mask = np.ones(len(matrix[2 * i]), dtype=bool)
        mask[anomalies] = False
        matrix

    return matrix


# calculates the RMS of each n*2 column in the input matrix and writes it to the n*2 + 1 column
def root_mean_square(matrix):
    return matrix


def slice_csv(subject_number: int):
    subject_number_str = str(subject_number)
    folder_path = "Data/EMG/Subject" + subject_number_str
    file_prefix_path = folder_path + "/subject" + subject_number_str + "_"
    json_path = file_prefix_path + "slices.json"

    slices_file = open(json_path)
    slices = json.load(slices_file)

    for round_name, slices_round in slices.items():
        csv_path = file_prefix_path + round_name + ".csv"
        output_folder_path = folder_path + "/sliced"
        os.makedirs(output_folder_path, exist_ok=True)

        with open(csv_path, mode="r", newline="", encoding="utf-8") as infile:
            csv_reader = csv.reader(infile, delimiter=";")
            rows = list(csv_reader)

            for exercise, slices_exercise in slices_round.items():
                output_path = (
                    output_folder_path + "/" + round_name + "_" + exercise + ".csv"
                )

                with open(
                    output_path, mode="a", newline="", encoding="utf-8"
                ) as outfile:
                    csv_writer = csv.writer(outfile)
                    csv_writer.writerows(rows[ROW_HEADER_START : ROW_HEADER_START + 1])
                    print(f"Header has been copied to {output_path}")

                    for s in slices_exercise:
                        selected_rows = np.array(
                            rows[s[0] + ROW_DATA_START - 1 : s[1] + ROW_DATA_START]
                        )

                        # removes falsely applied windowed RMS from Trigno Capture
                        selected_rows_no_rms = selected_rows[:, ::2]
                        print(selected_rows[0])
                        print(selected_rows_no_rms[0])

                        # rows_anomaly_corrected = remove_anomalies(selected_rows)
                        # rows_added_RMS = root_mean_square(rows_anomaly_corrected)

                        # csv_writer.writerows(rows_anomaly_corrected)

                        # print(
                        #    f"Rows {s[0]} to {s[1]} have been copied to {output_path}"
                        # )


for i in range(1, 2):
    slice_csv(i)
