import os
from PIL import Image
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt



def combine_masks(base_file, mask_files):

    im_frame = Image.open(mask_files[0])
    np_frame = np.array(im_frame)
    combined_mask = np.zeros(np_frame.shape)

    for mask in mask_files:
        im_frame = Image.open(mask)
        np_frame = np.array(im_frame)
        combined_mask += np_frame

    im = Image.fromarray(combined_mask.astype(np.uint8))
    im.save(base_file.replace(".JPG","_mask.png"))

folder = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\Code\\cc-machine-learning\\valid"

print(os.listdir(folder))

png_files = [f for f in os.listdir(folder) if f.endswith(".png")]
jpg_files = [f for f in os.listdir(folder) if f.endswith(".JPG")]
print(png_files)

for f in jpg_files:
    abs_base_file = os.path.join(folder, f)
    base_file_name = f.split(".")[0]
    assosiated_masks = [os.path.join(folder, p) for p in png_files if ("_").join(p.split("_")[:-1]) == base_file_name]
    print(base_file_name)
    combine_masks(abs_base_file, assosiated_masks)