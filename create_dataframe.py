import os
import pandas as pd
from PIL import Image
import numpy as np

CLASS_DICT = {	    "background" : 0,
                    "liverwort" : 1,
                    "moss" : 2,
                    "cyanosliverwort" : 3,
                    "cyanosmoss" : 4,
                    "lichen" : 5,
                    "barkdominated" : 6,
                    "cyanosbark" : 7,
                }

def read_images_to_dataframe(directory):
    data = []
    labels = []
    class_names = list(CLASS_DICT.keys())
    df = pd.DataFrame(columns=['file_name', 'forest_type', 'ground_level', 'direction', 'liverwort', 'moss', 'cyanosliverwort', 'cyanosmoss', 'lichen', 'barkdominated', 'cyanosbark'])

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            label = os.path.splitext(filename)[0]
            image = np.array(Image.open(image_path))
            file_codes = filename.split('_')
            forest_type = file_codes[1]
            ground_level = file_codes[2]
            direction = file_codes[3]
            
            class_counts = np.bincount(image.flatten())[1:]
            class_distributions = (class_counts/np.sum(class_counts))
            # print(class_distributions)
            df = df.append({'file_name': filename, 'forest_type': forest_type, 'ground_level': ground_level, 'direction': direction, 'liverwort': class_distributions[0], 'moss': class_distributions[1], 'cyanosliverwort': class_distributions[2], 'cyanosmoss': class_distributions[3], 'lichen': class_distributions[4], 'barkdominated': class_distributions[5], 'cyanosbark': class_distributions[6]}, ignore_index=True)

    return df


if __name__ == '__main__':

    path = "C:\\Users\\phili\\Documents\\Work\\PhD - Institut Biologie\\Results\\MaskPredictions"
    df = read_images_to_dataframe(path)
    # df.to_pickle('dataframe.pkl')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
