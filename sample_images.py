import os
import random
import shutil

data_path = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\Data\\ATTO\\Terra Firme"
training_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\training\\imgs"
test_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\test\\atto\\Feb"

if data_path.split("\\")[-1] == "Terra Firme":
    height_zones = ["Canopy", "Main_stem", "Ground"]
elif data_path.split("\\")[-1] == "Campina":
    height_zones = ["Canopy", "Ground"]
    
directions = ["East", "North", "South", "West"]

training_images = os.listdir(training_path)
img_count = 0
to_copy = 3

for height_zone in height_zones:
    for direction in directions:
        sub_label_path = os.path.join(data_path, height_zone, direction, "Sub_label")
        imgs = os.listdir(sub_label_path)
        reduced = [i for i in imgs if i not in training_images]
        
        num_copied = 0
        
        while num_copied < to_copy:
            sampled = random.sample(reduced, 1)[0]
            sampled_path = os.path.join(sub_label_path, sampled)
            reduced = [x for x in reduced if x != sampled]
            shutil.copy2(sampled_path, test_path)
            num_copied += 1
            print(sampled_path)
