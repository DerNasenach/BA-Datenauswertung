import pandas as pd

"""
helper file to unify format of two CAPTIV csv datafiles and aggregate them, 
allowing for easier access during evaluation
used for the aggregation of subject 5 without exo data, as one exercise was recorded separately,
resulting in two raw files with differing formats for the round without exo.
"""


def merge_captiv_datasets(
    base_csv_path,
    hoeherlagern_csv_path,
    output_csv_path,
):
    # Helper to find where the data starts
    def find_data_start(filepath):
        with open(filepath, encoding="utf-16") as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Time,"):
                    return i
        raise ValueError("No data header found")

    base_start = find_data_start(base_csv_path)
    base_df = pd.read_csv(base_csv_path, skiprows=base_start, encoding="utf-16")
    hoeher_start = find_data_start(hoeherlagern_csv_path)
    hoeher_df = pd.read_csv(
        hoeherlagern_csv_path, skiprows=hoeher_start, encoding="utf-16"
    )

    # Remove duplicate time columns
    cols = hoeher_df.columns
    keep_cols = [col for i, col in enumerate(cols) if i == 0 or (i % 2 == 1)]
    hoeher_df = hoeher_df[keep_cols]

    # Add missing trailing columns if exists in base_df
    for col in base_df.columns:
        if col not in hoeher_df.columns:
            hoeher_df[col] = pd.NA

    # Swap columns for 1551 and 1552, matching the 'Back' sensor
    def swap_1551_1552(col):
        if " 1551" in col:
            return col.replace(" 1551", " 1551_tmp")
        elif " 1552" in col:
            return col.replace(" 1552", " 1551")
        return col

    hoeher_df = hoeher_df.rename(columns=swap_1551_1552)
    hoeher_df = hoeher_df.rename(columns=lambda col: col.replace(" 1551_tmp", " 1552"))

    hoeher_df = hoeher_df[base_df.columns]
    merged_df = pd.concat([base_df, hoeher_df], ignore_index=True)

    # write to file
    with open(base_csv_path, encoding="utf-16") as f:
        lines = f.readlines()
    data_start = find_data_start(base_csv_path)
    header = "".join(lines[: data_start + 1])
    with open(output_csv_path, "w", encoding="utf-16") as f:
        f.write(header)
        merged_df.to_csv(f, index=False, header=False, lineterminator="\n")


if __name__ == "__main__":
    merge_captiv_datasets(
        "Data/CAPTIV/Subject5/subject5_ohne_exo_raw.csv",
        "Data/CAPTIV/Subject5/subject5_ohne_exo_hoeherlagern.csv",
        "Data/CAPTIV/Subject5/subject5_ohne_exo.csv",
    )
