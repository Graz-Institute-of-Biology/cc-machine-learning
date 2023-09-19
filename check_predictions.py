from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import random




def save_subplots(p, mask_frame, im_frame):
    fig, ax = plt.subplots(1,2, figsize=(19,10))
    ax[0].imshow(mask_frame,cmap=col, label=labels.keys())
    patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(colors))]
    ax[0].axis('off')
    ax[0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0. )    

    # plt.figure()
    ax[1].imshow(im_frame)
    ax[1].axis('off')

    fig.savefig(p, format='pdf')

def save_original(p, im_frame):
    fig = plt.figure(figsize=(19,10))
    plt.axis('off')

    plt.imshow(im_frame)

    fig.savefig(p, format='pdf')
    plt.close()

def save_mask(p, mask_frame, im_frame):

    fig = plt.figure(figsize=(19,10))
    plt.imshow(im_frame)
    plt.imshow(mask_frame,cmap=col, label=labels.keys(), alpha=0.8)
    patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(colors))]
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0. )   
    fig.savefig(p, format='pdf')
    plt.close()

size = 1024

data_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\training\\partial_images\\{0}".format(size)
pred_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\test\\{}\\pred".format(size)

orig_files = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])
pred_files = sorted([os.path.join(pred_path, x) for x in os.listdir(pred_path)])

labels = {	0:"background",
            1:"moss",
            2:"cyanosmoss",
            3:"lichen",
            4:"barkdominated",
            5:"cyanosbark",
            6:"other",
        }

left = np.array(range(20))
height = np.ones(20)

colors = ["#000000", "#00bcff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
col = ListedColormap(colors)

p = PdfPages("results_{0}.pdf".format(size))
SAMPLE = False
n = 100

if SAMPLE:
    orig_files = random.sample(orig_files, n)

for file_path in orig_files:
    file_name = file_path.split("\\")[-1]
    # print(file_name)
    im_frame = Image.open(file_path)
    try:
        pred_file_path = os.path.join(pred_path, file_name.replace(".JPG", "_OUT.png"))
        mask_frame = Image.open(pred_file_path)
        print("File found: {0}".format(pred_file_path))
    except FileNotFoundError:
        continue

    save_original(p, im_frame)
    save_mask(p, mask_frame, im_frame)

p.close()