def replace_delimiters(input_path):
    with open(input_path, "r") as file:
        content = file.read()
    content = content.replace(",", ".").replace(";", ",")

    with open(input_path, "w") as file:
        file.write(content)

    print(f"delimiters have been replaced in {input_path}")


for i in range(1, 9):
    replace_delimiters(
        "Data/EMG/Subject" + str(i) + "/subject" + str(i) + "_mit_exo.csv"
    )
    replace_delimiters(
        "Data/EMG/Subject" + str(i) + "/subject" + str(i) + "_ohne_exo.csv"
    )
