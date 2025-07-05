# replaces delimiters in EMG .csv files
# formating for easier read-in during the evaluation.


def replace_delimiters(input_path):
    with open(input_path, "r") as file:
        content = file.read()
    content = content.replace(",", ".").replace(";", ",")

    with open(input_path, "w") as file:
        file.write(content)

    print(f"delimiters have been replaced in {input_path}")


for i in range(1, 9):
    replace_delimiters(f"Data/EMG/Subject{i}/subject{i}_mit_exo.csv")
    replace_delimiters(f"Data/EMG/Subject{i}/subject{i}_ohne_exo.csv")
