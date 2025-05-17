import json

# Concatenates the slice indices for each exercise and appends them to the round

for i in range(1,9):
    json_path = 'Data/EMG/Subject' + str(i) + '/subject' + str(i) + '_slices.json'
    infile = json.load(open(json_path))
    for trial in infile.values():
        all_exercises = []
        for key, value in trial.items():
            if key.startswith("exercise"):
                all_exercises.extend(value)
        trial["exercises_concat"] = all_exercises
        with open(json_path, 'w') as f:
            json.dump(infile, f, indent=4)

