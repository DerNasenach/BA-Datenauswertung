import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

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
        data_clean = data[~np.isnan(data).any(axis=1)]

        for _, values in sensor_cols.items():
            cols = [values["qx"], values["qy"], values["qz"], values["qw"]]
            values["quaternions"] = data_clean[:, cols]

    return sensor_cols


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


if __name__ == "__main__":
    quats1 = extract_quaternions(
        "Data/CAPTIV/Subject4/config_ohne_exo.json",
        "Data/CAPTIV/Subject4/subject4_ohne_exo.csv",
    )
    metrics1 = compute_metrics(
        quats1["Pelvis"]["quaternions"], quats1["Back"]["quaternions"]
    )
    quats2 = extract_quaternions(
        "Data/CAPTIV/Subject4/config_mit_exo.json",
        "Data/CAPTIV/Subject4/subject4_mit_exo.csv",
    )
    metrics2 = compute_metrics(
        quats2["Pelvis"]["quaternions"], quats2["Back"]["quaternions"]
    )
    print(metrics1)
    print(metrics2)
    """
    parse_avt_to_json("Data/CAPTIV/Subject1/avatar_mit_exo.avt", "test.json")
    avatar_to_sensor_config("Data/CAPTIV")
    """
