import json

# Concatenates the slice indices for each exercise of a round and appends them to the round
# (not used in final analysis)

for i in range(1, 9):
    json_path = f"Data/EMG/Subject{i}/subject{i}_slices.json"
    infile = json.load(open(json_path))
    for trial in infile.values():
        all_exercises = []
        for key, value in trial.items():
            if key.startswith("exercise"):
                all_exercises.extend(value)
        trial["exercises_concat"] = all_exercises
        with open(json_path, "w") as f:
            json.dump(infile, f, indent=4)
