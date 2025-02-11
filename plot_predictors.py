import pandas as pd
import os

import matplotlib.pyplot as plt

def plot_predictors():
    # Read CSV file into DataFrame
    df = pd.read_csv('dist_dataframe_all_v11.csv')

    y_axis = "lichen"
    x_axis = "abs_altitude"

    # Plot two columns against each other
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_axis], df[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('{0} vs {1}'.format(y_axis, x_axis))
    plt.grid(True)
    plt.show()



def get_tree_info():
    # Read CSV file into DataFrame
    df = pd.read_csv('dist_dataframe_all_v11.csv')

    base_parent = "D:\\Mavic-3-Fotos\\ATTO"
    # parent_folder = 'C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\Neuer Ordner'
    tree_dict = {}
    for filename in df['source_file']:
        parent_folder = [os.path.join(base_parent,x) for x in os.listdir(base_parent) if filename.split("_")[0] in x][0]
        print(parent_folder)
        for root, dirs, files in os.walk(parent_folder):
            dji_name = filename.split("_")[-2] + "_" + filename.split("_")[-1] + ".JPG"
            print(dji_name)
            tree = "none"
            if dji_name in files:
                folder_list = root.split("\\")
                for folder in folder_list:
                    if "Tree" in folder:
                        print("TREE: ", folder)
                        tree = folder + "_" + filename.split("_")[0]
                tree_dict[filename] = tree
                break


    print(tree_dict)
    tree_df = pd.DataFrame.from_dict(tree_dict, orient='index')
    tree_df.to_csv("tree_dict.csv")
get_tree_info()