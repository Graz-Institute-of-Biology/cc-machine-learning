import os
from matplotlib import pyplot as plt
import numpy as np


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


def plot_csv():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    train_csv_path = "train_log.csv"
    val_csv_path = "valid_log.csv"

    df = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    print(np.mean(df["iou_score"][450:]))
    print(np.mean(df_val["iou_score"][450:]))

    plt.figure(figsize=(16,7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(np.arange(len(df["iou_score"])), df["iou_score"], label="train iou_score")
    plt.title("Optimization performance")
    plt.plot(np.arange(len(df_val["iou_score"])), df_val["iou_score"], label="validation iou_score")
    plt.xlabel("Epoch")
    plt.ylabel("IoU score")
    # ax.plot(np.arange(len(df["val_loss"])), df["val_loss"], label="val. loss")
    plt.legend()

    # ax[1].title.set_text("val. loss")
    # ax[2].plot(np.arange(len(df["mean_iou"])), df["mean_iou"])
    # ax[2].title.set_text("mean_iou")
    plt.show()


plot_csv()

