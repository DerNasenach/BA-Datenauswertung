import csv
import os
import json
import numpy as np

# slices the .csv data in the folders Data/EMG/Subject{n} per slice indices in Data/EMG/Subject{n}/subject{n}_slices.json

ROW_HEADER_START = 3
ROW_HEADER_END = 4
ROW_DATA_START = 7
VERBOSE = True


def slice_csv(subject_number: int):
    """
    slices the EMG .csv files for a subject according to the slice indices in the corresponding .json
    """
    subject_number_str = str(subject_number)
    folder_path = f"Data/EMG/Subject{subject_number_str}"
    file_prefix_path = folder_path + f"/subject{subject_number_str}_"
    json_path = f"{file_prefix_path}slices.json"

    slices_file = open(json_path)
    slices = json.load(slices_file)

    for round_name, slices_round in slices.items():
        csv_path = f"{file_prefix_path}{round_name}.csv"
        output_folder_path = f"{folder_path}/sliced"
        os.makedirs(output_folder_path, exist_ok=True)

        with open(csv_path, mode="r", newline="", encoding="utf-8") as infile:
            csv_reader = csv.reader(infile, delimiter=",")
            rows = list(csv_reader)

            for exercise, slices_exercise in slices_round.items():
                output_path = f"{output_folder_path}/{round_name}_{exercise}.csv"

                try:
                    os.remove(output_path)
                    if VERBOSE:
                        print(f"removed existing file {output_path}")
                except OSError:
                    pass

                with open(
                    output_path, mode="a", newline="", encoding="utf-8"
                ) as outfile:
                    csv_writer = csv.writer(outfile)
                    row_header = rows[ROW_HEADER_START]
                    row_header_trimmed = row_header[::2]

                    csv_writer.writerow(row_header_trimmed)

                    for s in slices_exercise:
                        selected_rows = np.array(
                            rows[s[0] + ROW_DATA_START - 1 : s[1] + ROW_DATA_START]
                        )

                        # keep every second row, removes falsely applied windowed RMS from Trigno Capture
                        selected_rows_no_rms = selected_rows[:, ::2]

                        csv_writer.writerows(selected_rows_no_rms)

                        if VERBOSE:
                            print(
                                f"Rows {s[0]} to {s[1]} have been copied to {output_path}"
                            )


if __name__ == "__main__":
    # main execution, slice all subjects .csv files
    for i in range(1, 9):
        slice_csv(i)
