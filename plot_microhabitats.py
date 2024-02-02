import pandas as pd
import matplotlib.pyplot as plt

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


SHORT_LABELS = ["Li", "Mo", "CLi", "CMo", "Lich", "Ba", "CBa"]
HEIGHTS = ["Ground", "Main stem", "Canopy"]
DIRECTIONS = ["North", "East", "South", "West"]

def get_dist_per_habitat(df, habitat_number):
    df_habitat = df[df['habitat_num'] == habitat_number]
    df_habitat = df_habitat.drop(['file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees'], axis=1)

    # print(df_habitat)
    # Calculate the mean of columns in the DataFrame
    column_means = df_habitat.mean()

    # Print the column means
    # print(column_means.sum())
    return column_means

def get_dist_per_height(df, height):
    df_height = df[df['height'] == height]
    df_height = df_height.drop(['file_name', 'forest_type', 'height', 'direction', 'habitat_num'], axis=1)
    column_means = df_height.mean()

    print(column_means)
    return column_means

def plot_heights(df):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    habitat_num = 0
    habitat_names = list(MICRO_HABITATS.keys())
    heights = ["G", "M", "C"]
    for j in range(3):
        column_means = get_dist_per_height(df, heights[j])
        print(column_means)
        axes[j].set_title(HEIGHTS[habitat_num])
        axes[j].bar(SHORT_LABELS, column_means.values)
        axes[j].set_ylim([0, 1])
        axes[j].set_ylabel('Class distribution [%]')
        axes[j].set_xlabel('Classes')

        habitat_num += 1

    plt.tight_layout()
    plt.show()

def get_dist_per_direction(df, direction):
    df_direction = df[df['direction'] == direction]
    df_direction = df_direction.drop(['file_name', 'forest_type', 'height', 'direction', 'habitat_num', 'rel_altitude', 'direction_degrees'], axis=1)
    column_means = df_direction.mean()

    print(column_means)
    return column_means

def plot_directions(df):
    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    habitat_num = 0
    habitat_names = list(MICRO_HABITATS.keys())
    directions = ["N", "E", "S", "W"]
    for j in range(len(directions)):
        column_means = get_dist_per_direction(df, directions[j])
        print(column_means)
        axes[j].set_title(DIRECTIONS[habitat_num])
        axes[j].bar(SHORT_LABELS, column_means.values)
        axes[j].set_ylim([0, 1])
        axes[j].set_ylabel('Class distribution [%]')
        axes[j].set_xlabel('Classes')

        habitat_num += 1

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
            axes[pos_h, pos_w].set_ylim([0, 1])
            axes[pos_h, pos_w].set_ylabel('Class distribution [%]')
            axes[pos_h, pos_w].set_xlabel('Classes')

            habitat_num += 1

    plt.tight_layout()
    plt.show()


file_path = "dist_dataframe_v9.csv"

df = pd.read_csv(file_path, index_col=0)
# print(df)

plot_all_micro_habitats(df)
# plot_heights(df)
# plot_directions(df)