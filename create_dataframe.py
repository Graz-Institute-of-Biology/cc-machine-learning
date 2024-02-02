import os
import pandas as pd
from PIL import Image
import numpy as np
import os
from PIL.ExifTags import TAGS

MICRO_HABITATS = {
                    "GN" : 0,
                    "GE" : 1,
                    "GS" : 2,
                    "GW" : 3,
                    "MN" : 4,
                    "ME" : 5,
                    "MS" : 6,
                    "MW" : 7,
                    "CN" : 8,
                    "CE" : 9,
                    "CS" : 10,
                    "CW" : 11,
}

CLASS_DICT = {	    "background" : 0,
                    "liverwort" : 1,
                    "moss" : 2,
                    "cyanosliverwort" : 3,
                    "cyanosmoss" : 4,
                    "lichen" : 5,
                    "barkdominated" : 6,
                    "cyanosbark" : 7,
                }

ORIG_DATA_DIR = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\Data\\ATTO"

def convert_string_to_float(number_string, degree_shift=False):
    """cast direction string to float
    from "+FLOAT" or "-FLOAT" to +float or -float
    Args:
        number_string (string): string with float and +/- sign

    Returns:
        float: converted float
    """
    if number_string[0] == "+":
        converted = float(number_string[1:])
    
    elif number_string[0] == "-":
        converted =  -float(number_string[1:])
        if degree_shift:
            converted += 360

    return converted

def get_direction_and_height(filename):
    filename = filename.split('_part')[0] + '.JPG'
    print(filename)
    for root, dirs, files in os.walk(ORIG_DATA_DIR):
        if filename in files:
            file_path = os.path.join(root, filename)
            img = Image.open(file_path)
            info = img._getexif()
            results = {}
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                results[decoded] = value
            direction_string = img.getxmp()['xmpmeta']['RDF']['Description']['GimbalYawDegree']
            direction_float = convert_string_to_float(direction_string, degree_shift=True)
            rel_altitude_string = img.getxmp()['xmpmeta']['RDF']['Description']['RelativeAltitude']
            rel_altitude_float = convert_string_to_float(rel_altitude_string)
            abs_altitude_string = img.getxmp()['xmpmeta']['RDF']['Description']['AbsoluteAltitude']
            abs_altitude_float = convert_string_to_float(abs_altitude_string)
            
    return direction_float, rel_altitude_float, abs_altitude_float


def read_images_to_dataframe(directory):
    data = []
    labels = []
    class_names = list(CLASS_DICT.keys())
    df = pd.DataFrame(columns=['source_file', 'file_name', 'forest_type', 'height', 'direction', 'rel_altitude', 'direction_degrees','habitat_num', 'liverwort', 'moss', 'cyanosliverwort', 'cyanosmoss', 'lichen', 'barkdominated', 'cyanosbark'])

    for filename in os.listdir(directory):
        if "_TF_" in filename and filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            label = os.path.splitext(filename)[0]
            image = np.array(Image.open(image_path))
            file_codes = filename.split('_')
            forest_type = file_codes[1]
            height = file_codes[2]
            direction = file_codes[3]
            habitat_number = MICRO_HABITATS[height + direction]
            source_file = filename.split("_part")[0]
            
            class_counts = np.bincount(image.flatten())[1:]
            class_distributions = np.zeros(len(class_names))
            class_distributions[:len(class_counts)] = (class_counts/np.sum(class_counts))
            # print(class_distributions)
            direction_degrees, rel_altitude, abs_altitude = get_direction_and_height(filename)
            
            row_dict = {
                        'source_file': source_file,
                        'file_name': filename,
                        'forest_type': forest_type,
                        'height': height,
                        'direction': direction,
                        'rel_altitude': rel_altitude,
                        'abs_altitude': abs_altitude,
                        'direction_degrees': direction_degrees, 
                        'habitat_num' : int(habitat_number),
                        'liverwort': class_distributions[0],
                        'moss': class_distributions[1],
                        'cyanosliverwort': class_distributions[2],
                        'cyanosmoss': class_distributions[3],
                        'lichen': class_distributions[4],
                        'barkdominated': class_distributions[5],
                        'cyanosbark': class_distributions[6]
                        }
            df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
            
    return df

def combine_subimages(df):
    df_combined = pd.DataFrame(columns=['source_file', 'file_name', 'forest_type', 'height', 'direction', 'rel_altitude', 'abs_altitude', 'direction_degrees','habitat_num', 'liverwort', 'moss', 'cyanosliverwort', 'cyanosmoss', 'lichen', 'barkdominated', 'cyanosbark'])
    for source_file in df['source_file'].unique():
        df_subimages = df[df['source_file'] == source_file]
        row_dict = {
                    'source_file': source_file,
                    'forest_type': df_subimages['forest_type'].iloc[0],
                    'height': df_subimages['height'].iloc[0],
                    'direction': df_subimages['direction'].iloc[0],
                    'rel_altitude': df_subimages['rel_altitude'].iloc[0],
                    'abs_altitude': df_subimages['abs_altitude'].iloc[0],
                    'direction_degrees': df_subimages['direction_degrees'].iloc[0],
                    'habitat_num' : df_subimages['habitat_num'].iloc[0],
                    'liverwort': df_subimages['liverwort'].mean(),
                    'moss': df_subimages['moss'].mean(),
                    'cyanosliverwort': df_subimages['cyanosliverwort'].mean(),
                    'cyanosmoss': df_subimages['cyanosmoss'].mean(),
                    'lichen': df_subimages['lichen'].mean(),
                    'barkdominated': df_subimages['barkdominated'].mean(),
                    'cyanosbark': df_subimages['cyanosbark'].mean()
                    }
        df_combined = pd.concat([df_combined, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
    return df_combined

if __name__ == '__main__':

    # path = "C:\\Users\\phili\\Documents\\Work\\PhD - Institut Biologie\\Results\\MaskPredictions"
    path = "C:\\Users\\faulhamm\\Documents\\Philipp\\training\\partial_masks\\Datasets\\dataset_v9_160imgs_moss_and_habitat_balance"
    df = read_images_to_dataframe(path)
    # df.to_pickle('dataframe.pkl')
    df.to_csv('dist_dataframe_tf_v9.csv')

    df_combined = combine_subimages(df)
    df_combined.to_csv('dist_dataframe_combined_v9.csv')
