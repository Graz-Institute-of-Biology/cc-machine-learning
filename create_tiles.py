from PIL import Image
from itertools import product
import os


def tile(filename, dir_in, dir_out, d):
    """generate image tiles from: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python

    Args:
        filename (_type_): _description_
        dir_in (_type_): _description_
        dir_out (_type_): _description_
        d (_type_): _description_
    """
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)



test_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\test\\full"
dir_out = "C:\\Users\\faulhamm\\Documents\\Philipp\\test\\1024\\samples"
tile_size = 1024

for filename in os.listdir(test_path):
    tile(filename, test_path, dir_out=dir_out, d=tile_size)
    print(filename)