import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_slurm_out():

    with open('slurm-801555.out', encoding='utf-8') as f:
        lines = f.readlines()

    max_epochs = 500
    epoch_losses = np.zeros(max_epochs)
    # for epoch in range(max_epochs):
    #     epoch_line = [f for f in lines if f.startswith("Epoch {0}/".format(epoch)) and "100%" in f]
    #     # print(epoch_line)
    #     if len(epoch_line) > 1:
    #         epoch_losses[epoch] = float(epoch_line[-1].split("=")[-1][:-2])

    #     print(epoch_losses[epoch])

    train_loss_vals = np.array([float(str(f.split("=")[-1].split("]")[0])) for f in lines if len(f.split("=")) > 1])
    print(train_loss_vals)
    dice_scores = np.array([float(f.split("score: ")[-1]) for f in lines if len(f.split("score: ")) > 1])
    print(dice_scores)

    # print(lines[0])

    fig,ax = plt.subplots(1,2)
    ax[0].plot(np.arange(len(train_loss_vals)), train_loss_vals)
    ax[1].plot(np.arange(len(dice_scores)), dice_scores)
    plt.show()

def plot_csv(train_df, val_df):

    train_values = train_df["iou_score"].values
    val_values = val_df["iou_score"].values

    plt.figure(figsize=(16,7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(np.arange(len(train_values)), train_values, label="train iou_score", color="blue", alpha=0.99)
    plt.title("Optimization performance")
    plt.plot(np.arange(len(val_values)), val_values, label="validation iou_score", color="orange", alpha=0.99)
    plt.xlabel("Epoch")
    plt.ylabel("IoU score")
    # ax.plot(np.arange(len(df["val_loss"])), df["val_loss"], label="val. loss")
    plt.legend()

    # ax[1].title.set_text("val. loss")
    # ax[2].plot(np.arange(len(df["mean_iou"])), df["mean_iou"])
    # ax[2].title.set_text("mean_iou")
    plt.show()


def plot_csv_error_std(train_values, val_values, train_std, val_std):

    # print(np.mean(df["iou_score"][450:]))
    # print(np.mean(df_val["iou_score"][450:]))

    plt.figure(figsize=(16,7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(np.arange(len(train_values)), train_values, label="train iou_score", color="blue", alpha=0.99)
    plt.errorbar(np.arange(len(train_values)), train_values, train_std, linestyle="none", marker="^", label="train iou_score", color="blue", alpha=0.3)
    plt.title("Optimization performance")
    plt.plot(np.arange(len(val_values)), val_values, label="validation iou_score", color="orange", alpha=0.99)
    plt.errorbar(np.arange(len(val_values)), val_values, val_std, linestyle="none", marker="^", label="val iou_score", color="orange", alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("IoU score")
    plt.grid()
    # ax.plot(np.arange(len(df["val_loss"])), df["val_loss"], label="val. loss")
    plt.legend()

    # ax[1].title.set_text("val. loss")
    # ax[2].plot(np.arange(len(df["mean_iou"])), df["mean_iou"])
    # ax[2].title.set_text("mean_iou")
    plt.show()


def plot_multiple_experiments():
    import os
    import pandas as pd

    # Define the root directory where your experiment subdirectories are located
    root_directory = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\results\\dataset_v7_119_images\\15_seeds"

    # Initialize empty dataframes for train and validation logs
    train_logs_df = pd.DataFrame()
    valid_logs_df = pd.DataFrame()
    train_experiment_dfs = []
    val_experiment_dfs = []

    # Iterate through subdirectories
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)

        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            # Look for train_log.csv and valid_log.csv files
            train_log_path = os.path.join(subdir_path, "train_log.csv")
            valid_log_path = os.path.join(subdir_path, "valid_log.csv")

            # Check if the files exist
            if os.path.exists(train_log_path):
                # Read train_log.csv into a dataframe and add an experiment column
                train_df = pd.read_csv(train_log_path)
                train_df['Experiment'] = subdir  # Label each row with the experiment name
                train_experiment_dfs.append(train_df)

            if os.path.exists(valid_log_path):
                # Read valid_log.csv into a dataframe and add an experiment column
                valid_df = pd.read_csv(valid_log_path)
                valid_df['Experiment'] = subdir  # Label each row with the experiment name
                val_experiment_dfs.append(valid_df)

    # Concatenate all experiment dataframes by columns (parallel storage)
    train_final_df = pd.concat(train_experiment_dfs, axis=1)
    val_final_df = pd.concat(val_experiment_dfs, axis=1)

    vert_concat = pd.concat(val_experiment_dfs)
    max_index = np.argmax(vert_concat["iou_score"].values)
    max_exp = vert_concat.iloc[max_index]["Experiment"]
    max_score = vert_concat.iloc[max_index]["iou_score"]
    print(max_exp)
    print(max_score)

    train_mean_iou = np.mean(train_final_df["iou_score"].values, axis=1)
    val_mean_iou = np.mean(val_final_df["iou_score"].values, axis=1)

    train_std_iou = np.std(train_final_df["iou_score"].values, axis=1)
    val_std_iou = np.std(val_final_df["iou_score"].values, axis=1)

    # mean_dice = np.mean(final_df["Dice_score"].values, axis=1)
    plot_csv_error_std(train_mean_iou, val_mean_iou, train_std_iou, val_std_iou)


if __name__ == "__main__":
    train_csv_path = "results/train_log.csv"
    val_csv_path = "results/valid_log.csv"

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    # plot_slurm_out()
    plot_multiple_experiments()
    # plot_csv(train_df, val_df)
