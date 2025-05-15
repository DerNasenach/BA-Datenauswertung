import matplotlib.pyplot as plt


def extract_column(file_path, n):
    data = []
    time = []
    with open(file_path, "r") as file:
        content = file.read().split("\n")
        name = content[16].split("\t")[n]

        for i in range(18, len(content) - 2, 2):
            line = content[i].split("\t")
            data.append(float(line[n]))
            time.append(float(line[0]))

    return name, data, time


name, data, time = extract_column("./Data/BKMP/Subject1/subject1_ohne_exo.acp", 11)
# name2, data2, time2 = extract_column("./3_ohne_exo.acp", 10)

plt.plot(time, data, label=name)
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Subject2 ohne exo")

plt.grid(True)
plt.legend()

plt.show()
# plt.(figsize=(8, 4))
