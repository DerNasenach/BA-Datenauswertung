import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime

TRANSLATION_MAP_USED = {
    "Dos": "Back",
    "Bassin": "Pelvis",
    "Cuisse_G": "Thigh_l",
    "Cuisse_D": "Thigh_r",
}
TRANSLATION_MAP_UNUSED = {
    "Pied_G": "Foot_l",
    "Pied_D": "Foot_r",
    "Bras_G": "UpperArm_l",
    "Bras_D": "UpperArm_r",
    "ABras_G": "Forearm_l",
    "ABras_D": "Forearm_r",
    "Mollet_G": "Calf_l",
    "Mollet_D": "Calf_r",
    "Tete": "Head",
}


# extracts sensor configuration from an avatar.avt file, writes to json
def parse_avt_to_json(input_path, output_path):
    mapping = {}
    with open(input_path, "r", encoding="utf-16") as file:
        content = file.readlines()
        for i, line in enumerate(content):
            line = line.replace("\x00", "").strip()
            if not line or line.lower() == "true":
                continue
            if line in TRANSLATION_MAP_USED:
                french_name = line
                sensor_id = content[i + 2].split()[1][3:]
                english_name = TRANSLATION_MAP_USED[french_name]
                mapping[sensor_id] = english_name

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(mapping, json_file, indent=4)


# parses all .avt files in all subdirectories of the specified directory into a config.json
def avatar_to_sensor_config(base_dir):
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(".avt"):
                    avt_path = os.path.join(subdir_path, file)
                    json_path = (os.path.splitext(avt_path)[0] + ".json").replace(
                        "avatar", "config"
                    )
                    parse_avt_to_json(avt_path, json_path)


def extract_quaternions(config_path, csv_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    sensor_cols = {bodypart: {"sensor_id": sid} for sid, bodypart in config.items()}
    with open(csv_path, "r", encoding="utf-16") as f:
        for line in f:
            if line.startswith("Time,"):
                header = line.strip().split(",")
                break

        for idx, col in enumerate(header):
            if " " in col:
                axis, sensor_id = col.split()
                if sensor_id in config:
                    for _, v in sensor_cols.items():
                        if v["sensor_id"] == sensor_id:
                            v[axis] = idx

        # read data to 2d np.array. data always ends with ',' - skip last column
        data = np.genfromtxt(f, delimiter=",", filling_values=np.nan)[:, :-1]

        # some sensors end their collection slightly early, skip end rows that contain missing data to keep dimensions equal
        # data_clean = data[~np.isnan(data).any(axis=1)]

        for _, values in sensor_cols.items():
            cols = [values["qx"], values["qy"], values["qz"], values["qw"]]
            values["quaternions"] = data[:, cols]

    return sensor_cols


# slices into a list of structure: list_of_rounds(with exo, without exo)[list_of_exercises[list_of_body_region[values]]]
def get_slices_of_subject(subject_number):
    subject_number_str = str(subject_number)
    folder_path = f"Data/CAPTIV/Subject{subject_number_str}"
    file_prefix_path = f"{folder_path}/subject{subject_number_str}_"
    json_path = f"{folder_path}/subject{subject_number_str}_slices.json"
    slices_file = open(json_path)
    slices_indices = json.load(slices_file)

    config_ohne = f"{folder_path}/subject{subject_number_str}_config_ohne_exo.json"
    csv_ohne = f"{folder_path}/subject{subject_number_str}_ohne_exo.csv"
    config_mit = f"{folder_path}/subject{subject_number_str}_config_mit_exo.json"
    csv_mit = f"{folder_path}/subject{subject_number_str}_mit_exo.csv"

    quats_ohne = extract_quaternions(config_ohne, csv_ohne)
    quats_mit = extract_quaternions(config_mit, csv_mit)

    subject_slices = []

    for round_name, indices_round in slices_indices.items():
        round_slices = []
        if round_name == "ohne_exo":
            quats = quats_ohne
        elif round_name == "mit_exo":
            quats = quats_mit
        else:
            continue

        pelvis_quats = quats["Pelvis"]["quaternions"]
        back_quats = quats["Back"]["quaternions"]

        for exercise_name, indices_exercise in indices_round.items():
            # Collect all intervals for this exercise
            slices_pelvis = []
            slices_back = []
            for i_start, i_end in indices_exercise:
                slices_pelvis.append(pelvis_quats[i_start:i_end])
                slices_back.append(back_quats[i_start:i_end])

            slice_pelvis = np.concatenate(slices_pelvis, axis=0)
            slice_back = np.concatenate(slices_back, axis=0)
            round_slices.append((slice_pelvis, slice_back))

        subject_slices.append(round_slices)
    return subject_slices


def get_slices_all_subjects():
    slices = []
    for i in range(1, 9):
        slices.append(get_slices_of_subject(i))
    return slices


def compute_metrics(quaternions_0, quaternions_1):
    # Rotation Object from quaternion
    r_q0 = Rotation.from_quat(quaternions_0)
    r_q1 = Rotation.from_quat(quaternions_1)

    # Relative rotation ("Normalize". "Undo" rotation of q0 to set 0 point, match q1 to relative coordinate frame)
    r_relative = r_q1 * r_q0.inv()

    # Convert to Euler angles, pitch is X, yaw is Y
    euler = r_relative.as_euler("xyz", degrees=True)
    flexion = euler[:, 0]
    torsion = np.abs(euler[:, 1])

    # print(flexion)
    extension = -np.minimum(flexion, 0)
    flexion = np.maximum(flexion, 0)

    # Euclidian Norm
    general_angle = np.linalg.norm(euler, axis=1)

    metrics = {
        "max_torsion": np.max(torsion),
        "mean_torsion": np.mean(torsion),
        "max_flexion": np.max(flexion),
        "mean_flexion": np.mean(flexion),
        "max_extension": np.max(extension),
        "mean_extension": np.mean(extension),
        "max_total_angle": np.max(general_angle),
        "mean_total_angle": np.mean(general_angle),
    }
    return metrics


def make_eval_over_exercises(slices):
    NUMBER_OF_SUBJECTS = len(slices)
    NUMBER_OF_EXERCISES = len(slices[0][0])
    METRIC_NAMES = [
        "max_torsion",
        "mean_torsion",
        "max_flexion",
        "mean_flexion",
        "max_extension",
        "mean_extension",
        "max_total_angle",
        "mean_total_angle",
    ]

    reports = [
        {
            f"Exercise {ex+1}": {
                f"Subject {subj+1}": {
                    "ohne exo": {},
                    "mit exo": {},
                }
                for subj in range(NUMBER_OF_SUBJECTS)
            }
        }
        for ex in range(NUMBER_OF_EXERCISES)
    ]

    for subj_idx, subject in enumerate(slices):
        for round_idx, round_slices in enumerate(subject):
            round_name = "ohne exo" if round_idx == 0 else "mit exo"
            for ex_idx, (pelvis, back) in enumerate(round_slices):
                metrics = compute_metrics(pelvis, back)
                reports[ex_idx][f"Exercise {ex_idx+1}"][f"Subject {subj_idx+1}"][
                    round_name
                ] = metrics

    for ex_idx in range(NUMBER_OF_EXERCISES):
        agg = {"ohne exo": {}, "mit exo": {}, "percent_difference": {}}
        means = {"ohne exo": {}, "mit exo": {}}
        for round_name in ["ohne exo", "mit exo"]:
            vals = {k: [] for k in METRIC_NAMES}
            for subj_idx in range(NUMBER_OF_SUBJECTS):
                m = reports[ex_idx][f"Exercise {ex_idx+1}"][
                    f"Subject {subj_idx+1}"
                ].get(round_name, {})
                for k in METRIC_NAMES:
                    if k in m:
                        vals[k].append(m[k])
            for k in METRIC_NAMES:
                if vals[k]:
                    mean_val = float(np.mean(vals[k]))
                    max_val = float(np.max(vals[k]))
                    agg[round_name][f"mean_{k}"] = mean_val
                    agg[round_name][f"max_{k}"] = max_val
                    means[round_name][k] = mean_val
        # Calculate percent difference for each metric
        for k in METRIC_NAMES:
            mean_ohne = means["ohne exo"].get(k, None)
            mean_mit = means["mit exo"].get(k, None)
            if mean_ohne is not None and mean_ohne != 0 and mean_mit is not None:
                percent_diff = 100 * (mean_mit - mean_ohne) / abs(mean_ohne)
                agg["percent_difference"][k] = percent_diff
            else:
                agg["percent_difference"][k] = None
        reports[ex_idx][f"Exercise {ex_idx+1}"]["All subjects"] = agg

        os.makedirs("Data/CAPTIV/evaluations", exist_ok=True)
        with open(
            f"Data/CAPTIV/evaluations/evaluations_exercise_{ex_idx+1}.json", "w"
        ) as f:
            json.dump(reports[ex_idx], f, indent=4)


def make_eval_concat(slices):
    NUMBER_OF_SUBJECTS = len(slices)
    METRIC_NAMES = [
        "max_torsion",
        "mean_torsion",
        "max_flexion",
        "mean_flexion",
        "max_extension",
        "mean_extension",
        "max_total_angle",
        "mean_total_angle",
    ]
    report_concat = {"concatenated evaluation": {}}

    for subj_idx, subject in enumerate(slices):
        pelvis_ohne = np.concatenate([ex[0] for ex in subject[0]], axis=0)
        back_ohne = np.concatenate([ex[1] for ex in subject[0]], axis=0)
        pelvis_mit = np.concatenate([ex[0] for ex in subject[1]], axis=0)
        back_mit = np.concatenate([ex[1] for ex in subject[1]], axis=0)

        metrics_ohne = compute_metrics(pelvis_ohne, back_ohne)
        metrics_mit = compute_metrics(pelvis_mit, back_mit)

        report_concat["concatenated evaluation"][f"Subject {subj_idx+1}"] = {
            "ohne exo": metrics_ohne,
            "mit exo": metrics_mit,
        }

    agg = {"ohne exo": {}, "mit exo": {}, "percent_difference": {}}
    means = {"ohne exo": {}, "mit exo": {}}
    for round_name in ["ohne exo", "mit exo"]:
        vals = {k: [] for k in METRIC_NAMES}
        for subj_idx in range(NUMBER_OF_SUBJECTS):
            m = report_concat["concatenated evaluation"][f"Subject {subj_idx+1}"][
                round_name
            ]
            for k in METRIC_NAMES:
                if k in m:
                    vals[k].append(m[k])
        for k in METRIC_NAMES:
            if vals[k]:
                mean_val = float(np.mean(vals[k]))
                max_val = float(np.max(vals[k]))
                agg[round_name][f"mean_{k}"] = mean_val
                agg[round_name][f"max_{k}"] = max_val
                means[round_name][k] = mean_val
    # Calculate percent difference for each metric
    for k in METRIC_NAMES:
        mean_ohne = means["ohne exo"].get(k, None)
        mean_mit = means["mit exo"].get(k, None)
        if mean_ohne is not None and mean_ohne != 0 and mean_mit is not None:
            percent_diff = 100 * (mean_mit - mean_ohne) / abs(mean_ohne)
            agg["percent_difference"][k] = percent_diff
        else:
            agg["percent_difference"][k] = None
    report_concat["concatenated evaluation"]["All Subjects total"] = agg

    os.makedirs("Data/CAPTIV/evaluations", exist_ok=True)
    with open("Data/CAPTIV/evaluations/evaluations_concat.json", "w") as f:
        json.dump(report_concat, f, indent=4)


# used to make slices-indices from raw CAPTIV-Data
def calculate_index(
    start_time, end_time, frequency=32, time_format="%H:%M:%S.%f", offset=0
):
    start_dt = datetime.strptime(start_time, time_format)
    end_dt = datetime.strptime(end_time, time_format)
    return int((end_dt - start_dt).total_seconds() * frequency + offset)


if __name__ == "__main__":
    """
    timestamp_start_round = "16:07:13.113"
    timestamp_current = "16:09:48.207"
    print(calculate_index(timestamp_start_round, timestamp_current))
    avatar_to_sensor_config("Data/CAPTIV")
    """
    slices = get_slices_all_subjects()
    make_eval_over_exercises(slices)
    make_eval_concat(slices)
