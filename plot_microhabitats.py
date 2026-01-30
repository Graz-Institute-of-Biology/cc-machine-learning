import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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

colors_hex = ["#000000",
        "#1cffbb",
        "#00bcff",
        "#0059ff",
        "#2601d8",
        "#ff00c3",
        "#Ff0000",
        "#FFA500",
        "#FFFF00"]


colors_hex = colors_hex[::-1]

SHORT_LABELS = ["Liv.", "Moss", "CyL.", "CyM.", "Lich.", "Bark", "CyBark"]
SHORT_LABELS = SHORT_LABELS[::-1]

HEIGHTS = ["Canopy", "Main stem", "Ground"]
# HEIGHTS = ["Ground", "Canopy"]
# HEIGHTS = ["Canopy", "Ground"]

DIRECTIONS = ["North", "East", "South", "West"]

def get_dist_per_habitat(df, habitat_number):
    df_habitat = df[df['habitat_num'] == habitat_number]
    df_habitat = df_habitat.drop(['source_file', 'file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees', 'abs_altitude'], axis=1)

    # print(df_habitat)
    # Calculate the mean of columns in the DataFrame
    column_means = df_habitat.mean()*100 # get mean values in percent

    # Print the column means
    # print(column_means.sum())
    return column_means

def get_dist_per_height(df, height):
    df_height = df[df['height'] == height]
    df_height = df_height.drop(['source_file', 'file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees', 'abs_altitude'], axis=1)
    column_means = df_height.mean()*100 # get mean values in percent
    img_counts = len(df_height)

    return column_means, img_counts

def plot_heights(df, title):
    plt.rcParams.update({'font.size': 12})


    habitat_names = list(MICRO_HABITATS.keys())

    if "terra" in title.lower():
        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        heights = ["C", "M", "G"]
        positions = [0, 1, 2]

    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        heights = ["C", "G"]
        positions = [0, 1]

    fig.suptitle(title)
    habitat_num = 0
    
    for j in positions:
        print("Height: ", heights[j], "\n")
        column_means, img_counts = get_dist_per_height(df, heights[j])
        print("Img counts: ", img_counts)
        column_means = column_means[::-1]
        print(column_means)
        print(HEIGHTS)
        title_string = HEIGHTS[habitat_num] + " (n=" + str(img_counts) + ")"
        axes[j].set_title(title_string)
        axes[j].barh(SHORT_LABELS, column_means, color=colors_hex[1:])
        axes[j].set_xlim([0, 100])
        # axes[j].set_ylim([0, 100])
        axes[j].set_xlabel('Class distribution [%]', fontsize=20)
        axes[j].set_ylabel('Classes', fontsize=20)
        axes[j].grid()
        for i in range(len(column_means)):
            axes[j].annotate("{:.2f}%".format(column_means[i]), xy=(column_means[i]+7, SHORT_LABELS[i]), ha='center', va='center')

        habitat_num += 1

    plt.tight_layout()
    plt.show()

def get_dist_per_direction(df, direction):
    """ calculate the distribution of classes for a given direction

    Args:
        df (pandas dataframe): pandas dataframe containing the data
        direction (str): direction info [N, E, S, W]

    Returns:
        numpy array: mean values of the classes in percent (0-100)
    """
    df_direction = df[df['direction'] == direction]
    df_direction = df_direction.drop(['source_file', 'file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees', 'abs_altitude'], axis=1)
    column_means = df_direction.mean()*100 # get mean values in percent

    print(column_means)
    return column_means

def plot_directions(df):
    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    plt.rcParams.update({'font.size': 10})
    fig.suptitle(title, fontsize=30)


    habitat_num = 0
    habitat_names = list(MICRO_HABITATS.keys())
    directions = ["N", "E", "S", "W"]
    for j in range(len(directions)):
        column_means = get_dist_per_direction(df, directions[j])
        print(column_means)
        axes[j].set_title(DIRECTIONS[habitat_num])
        axes[j].bar(SHORT_LABELS, column_means.values, color=colors_hex[1:])
        axes[j].set_ylim([0, 100])
        axes[j].set_ylabel('Class distribution [%]', fontsize=20)
        axes[j].set_xlabel('Classes', fontsize=20)

        for i in range(len(column_means)):
            axes[j].annotate("{:.2f}%".format(column_means[i]), xy=(SHORT_LABELS[i],column_means[i]), ha='center', va='bottom')

        habitat_num += 1
        axes[j].grid()

    plt.tight_layout()
    plt.show()

def plot_all_micro_habitats(df):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    habitat_num = 0
    habitat_names = list(MICRO_HABITATS.keys())
    for i in range(3):
        for j in range(4):
            pos_h = 2-i
            pos_w = j
            column_means = get_dist_per_habitat(df, habitat_num)
            print(habitat_names[habitat_num])
            print(column_means)
            axes[pos_h, pos_w].set_title(habitat_names[habitat_num])
            axes[pos_h, pos_w].bar(SHORT_LABELS, column_means.values, color=colors_hex[1:])
            axes[pos_h, pos_w].set_ylim([0, 100])
            axes[pos_h, pos_w].set_ylabel('Class distribution [%]')
            axes[pos_h, pos_w].set_xlabel('Classes')

            habitat_num += 1
            axes[pos_h, pos_w].grid()

    plt.tight_layout()
    plt.show()


def plot_overall_(df, title):


    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    plt.rcParams.update({'font.size': 30})
    fig.suptitle(title)
    df = df.drop(['source_file', 'file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees', 'abs_altitude'], axis=1)
    column_means = df.mean()*100
    axes.set_ylim([0, 100])
    
    # matplotlib.rc('xtick', labelsize=30) 
    # matplotlib.rc('ytick', labelsize=30)
    # matplotlib.rc('axes', titlesize=30)
    # matplotlib.rc('axes', labelsize=30)

    plt.xlabel('Classes', fontsize=30)
    plt.ylabel('Class distribution [%]', fontsize=30)
    axes.yaxis.set_tick_params(labelsize='small')
    axes.xaxis.set_tick_params(labelsize='small')

    axes.bar(SHORT_LABELS, column_means.values, color=colors_hex[1:])
    for i in range(len(column_means)):
        plt.annotate("{:.2f}%".format(column_means[i]), xy=(SHORT_LABELS[i],column_means[i]), ha='center', va='bottom')

    plt.tight_layout()
    plt.grid()
    plt.show()


# file_path = "dist_dataframe_dummy.csv"
# title = "Coverage distribution of example image"

file_path_c = "dist_dataframe_c_v11full.csv"
title_c = "Coverage distribution across heights in Campina"

file_path_tf = "dist_dataframe_tf_v11full.csv"
title_tf = "Coverage distribution across heights of Terra Firme"

df_tf = pd.read_csv(file_path_tf, index_col=0)
df_c = pd.read_csv(file_path_c, index_col=0)
print(len(df_tf))
# print(df)

# plot_all_micro_habitats(df)
plot_heights(df_tf, title=title_tf)
plot_heights(df_c, title=title_c)
# plot_directions(df)
# plot_overall_(df, title=title)