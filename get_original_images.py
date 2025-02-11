

import os
import shutil

entropy_dir = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\results\\dataset_v11_C\\Entropies_Sub"
# data_dir = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\Data\\ATTO\\Campina"
data_dir = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\results\\dataset_v11_C\\Entropies_280824"


file_list = [x.replace("_entropy", "_mask") for x in os.listdir(entropy_dir)]
# print(file_list)

for dirpaths, dirnames, files in os.walk(data_dir):
    for file in files:
        # print(file)
        if file in file_list:
            print("FOUND: ", file)
            print(os.path.join(dirpaths, file))
            shutil.copy2(os.path.join(dirpaths, file), os.path.join(entropy_dir, file))
            
    