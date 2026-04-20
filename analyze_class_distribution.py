import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# DATASETS_ROOT = Path("C:/Users/faulhamm/Documents/Philipp/training/datasets/ATTO") ## ATTO
# ONTOLOGY_PATH = Path(__file__).parent / "ontology_atto.json"

DATASETS_ROOT = Path("C:\\Users\\faulhamm\\Documents\\Philipp\\training\\grossglockner\\saved_datasets") ## GROSSGLOCKNER
ONTOLOGY_PATH = Path(__file__).parent / "ontology_gg.json"

with open(ONTOLOGY_PATH) as f:
    ontology = json.load(f)["ontology"]

classes = {name: entry["value"] for name, entry in ontology.items()}
class_names = list(classes.keys())
class_values = list(classes.values())
class_colors = [entry["color"] for entry in ontology.values()]

datasets = sorted([d for d in DATASETS_ROOT.iterdir() if d.is_dir()])

all_counts = []
for dataset in datasets:
    mask_dir = dataset / "partial_masks"
    mask_paths = list(mask_dir.glob("*.png"))
    counts = np.zeros(len(class_values), dtype=np.int64)
    for mask_path in tqdm(mask_paths, desc=dataset.name, unit="mask"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        for idx, val in enumerate(class_values):
            counts[idx] += np.sum(mask == val)
    all_counts.append(counts)

    # plot & save into dataset folder
    total = counts.sum()
    pcts = [100 * counts[i] / total if total > 0 else 0 for i in range(len(class_values))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, pcts, color=class_colors, edgecolor="black", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        if pct > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Pixel %")
    ax.set_title(dataset.name)
    ax.set_ylim(0, max(pcts) * 1.15 + 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(dataset / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved plot to {dataset / 'class_distribution.png'}")

# summary table
col_w = 20
header = f"{'Class':<{col_w}}" + "".join(f"{d.name:>26}" for d in datasets)
print("\n" + header)
print("-" * len(header))
for class_idx, class_name in enumerate(class_names):
    row = f"{class_name:<{col_w}}"
    for counts in all_counts:
        total = counts.sum()
        pct = 100 * counts[class_idx] / total if total > 0 else 0
        row += f"{pct:>25.2f}%"
    print(row)
print("-" * len(header))
total_row = f"{'total pixels':<{col_w}}"
for counts in all_counts:
    total_row += f"{counts.sum():>26,}"
print(total_row)
