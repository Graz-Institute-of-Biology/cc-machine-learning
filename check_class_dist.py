import numpy as np
import cv2
import os
import json
import tqdm

class ClassDistributionChecker:

    def __init__(self, dataset_path, ontology_file="ontology_atto.json", n_folds=5, cross_val_run=0, seed=10):
        self.img_dir = dataset_path + "/partial_images/"
        self.mask_dir = dataset_path + "/partial_masks/"
        self.ontology_file = ontology_file
        self.n_folds = n_folds
        self.cross_val_run = cross_val_run
        self.seed = seed

        np.random.seed(self.seed)
        self.ids = sorted(os.listdir(self.img_dir))

        self.load_ontology()
        self.class_values = [d["value"] for d in self.ontology["ontology"].values()]
        self.labels = list(self.ontology["ontology"].keys())

        self.set_non_unique_paths()


    def load_ontology(self):
        # Load ontology from a predefined path or dictionary
        ontology_path = self.ontology_file
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)

    def check_class_distribution(self):

        print("\n=== Class Distribution Analysis ===")
        train_class_counts = np.zeros(len(self.class_values))
        val_class_counts = np.zeros(len(self.class_values))
        
        # Count training set
        print("Counting training set class distribution...")
        for mask_path in tqdm.tqdm(self.y_train):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            for idx, class_val in enumerate(self.class_values):
                train_class_counts[idx] += np.sum(mask == class_val)
        
        # Count validation set
        print("Counting validation set class distribution...")
        for mask_path in tqdm.tqdm(self.y_valid):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            for idx, class_val in enumerate(self.class_values):
                val_class_counts[idx] += np.sum(mask == class_val)
        
        # Calculate percentages
        train_total = train_class_counts.sum()
        val_total = val_class_counts.sum()
        
        print(f"{'Class':<20} {'Train %':<12} {'Val %':<12} {'Ratio (T/V)':<12}")
        print("-" * 60)
        for idx, label in enumerate(self.labels):
            train_pct = 100 * train_class_counts[idx] / train_total if train_total > 0 else 0
            val_pct = 100 * val_class_counts[idx] / val_total if val_total > 0 else 0
            ratio = train_pct / val_pct if val_pct > 0 else float('inf')
            print(f"{label:<20} {train_pct:>6.2f}%     {val_pct:>6.2f}%     {ratio:>6.2f}x")
        print("=" * 60 + "\n")


    def set_non_unique_paths(self, train_split=0.8):

        self.images_fps = sorted([os.path.join(self.img_dir, image_id) for image_id in self.ids if image_id.lower().endswith(".jpg")])
        if self.images_fps[0].endswith(".JPG"):
            self.masks_fps = sorted([os.path.join(self.mask_dir, image_id.replace("JPG", "png")) for image_id in self.ids])
        elif self.images_fps[0].endswith(".jpg"):
            self.masks_fps = sorted([os.path.join(self.mask_dir, image_id.replace("jpg", "png")) for image_id in self.ids])


        train_len = int(len(self.images_fps) * train_split)

        indices = np.random.choice(np.arange(len(self.images_fps)), len(self.images_fps))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        self.x_train = [self.images_fps[index] for index in train_indices]
        self.y_train = [self.masks_fps[index] for index in train_indices]

        self.x_valid = [self.images_fps[index] for index in val_indices]
        self.y_valid = [self.masks_fps[index] for index in val_indices]

        self.x_test = self.x_valid
        self.y_test = self.y_valid

    def set_paths(self):

        # get unique image names
        original_imgs = [x.split("_part")[0] for x in self.ids]
        unique_imgs = sorted(np.unique(original_imgs))
        print(unique_imgs)
        if len(unique_imgs) == 0:
            print("Error: No images found!")
            return None

        shuffled_original = unique_imgs.copy()
        np.random.shuffle(shuffled_original)

        val_fold_size = len(unique_imgs) // self.n_folds
        val_start = self.cross_val_run * val_fold_size
        val_end = val_start + val_fold_size if self.cross_val_run < self.n_folds - 1 else len(unique_imgs)
        
        unique_val_imgs = shuffled_original[val_start:val_end]
        unique_train_imgs = shuffled_original[:val_start] + shuffled_original[val_end:]
        print("Val start: ", val_start)
        print("Val end: ", val_end)
        print("Train images: ", len(unique_train_imgs))
        print("Val images: ", len(unique_val_imgs))


        self.x_train = [os.path.join(self.img_dir, x) for x in self.ids if x.split("_part")[0] in unique_train_imgs]

        if self.x_train[0].endswith(".JPG"):
            self.y_train = [os.path.join(self.mask_dir, x.replace("JPG", "png")) for x in self.ids if x.split("_part")[0] in unique_train_imgs]
        elif self.x_train[0].endswith(".jpg"):
            self.y_train = [os.path.join(self.mask_dir, x.replace("jpg", "png")) for x in self.ids if x.split("_part")[0] in unique_train_imgs]
        
        self.x_valid = [os.path.join(self.img_dir, x) for x in self.ids if x.split("_part")[0] in unique_val_imgs]

        if self.x_valid[0].endswith(".JPG"):
            self.y_valid = [os.path.join(self.mask_dir, x.replace("JPG", "png")) for x in self.ids if x.split("_part")[0] in unique_val_imgs]
        elif self.x_valid[0].endswith(".jpg"):
            self.y_valid = [os.path.join(self.mask_dir, x.replace("jpg", "png")) for x in self.ids if x.split("_part")[0] in unique_val_imgs]

        self.x_test = self.x_valid.copy()
        self.y_test = self.y_valid.copy()


    def calculate_class_weights(self):
        # Calculate class weights based on pixel frequency
        # Add this before: self.focal_loss = smp.losses.FocalLoss(...)

        # Get class pixel counts from your training set
        class_pixel_counts = np.zeros(len(self.class_values))
        for mask_path in self.y_train:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            for idx, class_val in enumerate(self.class_values):
                class_pixel_counts[idx] += np.sum(mask == class_val)

        # Calculate inverse frequency weights
        total_pixels = class_pixel_counts.sum()
        class_weights = np.zeros(len(self.class_values))

        for idx in range(len(self.class_values)):
            if class_pixel_counts[idx] > 0:
                # Inverse frequency
                class_weights[idx] = total_pixels / (len(self.class_values) * class_pixel_counts[idx])
            else:
                # Handle classes with 0 pixels (like "other")
                class_weights[idx] = 0.0

        # Normalize weights so they average to 1.0
        non_zero_mask = class_weights > 0
        if non_zero_mask.any():
            mean_weight = class_weights[non_zero_mask].mean()
            class_weights = class_weights / mean_weight

        # Set background weight to 0 since we're ignoring it
        class_weights[0] = 0.0

        print("\n=== Class Weights ===")
        for idx, label in enumerate(self.labels):
            print(f"{label:<20} Weight: {class_weights[idx]:.2f}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    dataset_path = r"C:\Users\faulhamm\Documents\Philipp\training\datasets\ATTO\dataset_v11_0_TF_split"  # Update with actual path

    checker = ClassDistributionChecker(dataset_path)
    checker.check_class_distribution()
    checker.calculate_class_weights()