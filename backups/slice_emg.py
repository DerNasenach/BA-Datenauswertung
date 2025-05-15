import json
import pandas as pd
import numpy as np

file = open('Data/EMG/Subject1/subject1_slices.json')
slices = json.load(file)
#for key, value in slices.items():
#    print(key)
#for s in slices['ohne_exo']['exercise4']:
#    print(s)


def slice_csv(subject_number: int):
    subject_number_str = str(subject_number)
    folder_path = 'Data/EMG/Subject' + subject_number_str
    file_prefix_path = folder_path + '/subject' + subject_number_str + '_'
    json_path = file_prefix_path + 'slices.json'

    slices_file = open(json_path) 
    slices = json.load(slices_file)

    for round_name, sclices_round in slices.items():
        csv_path = file_prefix_path + round_name + '.csv'
        print(csv_path)
        data = np.genfromtxt(
            csv_path, 
            delimiter=";",
            dtype=str,
            skip_header=3
        )
        print(data[0])
        #data = pd.read_csv(file_prefix_path + round_name + '.csv', sep=';', skiprows=range(0,7), index_col=None)
        #print(round_name)
        #print(sclices_round)
       # print(data.iloc[0])


slice_csv(1)