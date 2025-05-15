import csv
import os
import numpy as np

ROW_HEADER_START = 3
ROW_HEADER_END = 7


def copy_header(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = np.genfromtxt(
        input_path,
        delimiter=";",
        dtype=str,
        skip_header=ROW_HEADER_START,
        max_rows=ROW_HEADER_END - ROW_HEADER_START,
    )
    np.savetxt(
        output_path,
        data,
        delimiter=";",
        fmt="%s",
        header=";".join(data[0]),
        comments="",
    )

    print(f"Header has been copied to {output_path}")


def copy_rows(input_path, output_path, start, end):

    data = np.genfromtxt(input_path, delimiter=";", dtype=str)
    selected_rows = data[start - 1 : end]
    with open(output_path, "a", newline="", encoding="utf-8") as outfile:
        np.savetxt(outfile, selected_rows, delimiter=";", fmt="%s")

    print(f"Rows {start} to {end} have been copied to {output_path}")


def replace_comma(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
    content = content.replace(",", ".")

    with open(input_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"commas have been replaced with periods in {input_path}")


slices = []


"""
source_folder = "./Data/Subject2/EMG/"
source_file = "ohne_exo"
source_path = source_folder + source_file + ".csv"
# start and end manually extracted from trigno discover review
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise1.csv", [(69112, 90830)])
)
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise2.csv", [(96115, 129841)])
)
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise3.csv", [(135834, 176133)])
)
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise4.csv", [(182728, 212007)])
)
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise5.csv", [(218452, 271898)])
)
slices.append(
    (source_folder + "sliced/" + source_file + "_exercise6.csv", [(374343, 513199)])
)
slices.append(
    (
        source_folder + "sliced/" + source_file + "_exercises_combined.csv",
        [
            (69112, 90830),
            (96115, 129841),
            (135834, 176133),
            (182728, 212007),
            (218452, 271898),
            (374343, 513199),
        ],
    )
)
"""

source_folder = "./Data/Subject2/EMG/"
source_file = "mit_exo"
source_path = source_folder + source_file + ".csv"
# start and end manually extracted from trigno discover review
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise1.csv", [(64408, 80261)])
)
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise2.csv", [(84622, 97747)])
)
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise3.csv", [(101399, 136865)])
)
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise4.csv", [(142708, 168078)])
)
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise5.csv", [(173255, 220492)])
)
slices.append(
    (source_folder + "slicedtest/" + source_file + "_exercise6.csv", [(226872, 355203)])
)
slices.append(
    (
        source_folder + "slicedtest/" + source_file + "_exercises_combined.csv",
        [
            (64408, 80261),
            (84622, 97747),
            (101399, 136865),
            (142708, 168078),
            (173255, 220492),
            (226872, 355203),
        ],
    )
)


for s in slices:
    copy_header(source_path, s[0])
    for rows in s[1]:
        copy_rows(source_path, s[0], rows[0], rows[1])
    replace_comma(s[0])
