import subprocess
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from smp_dataset import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
# import albumentations as albus
import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics as smp_metrics
import ssl
import random
from tqdm import tqdm
from pathlib import Path
import yaml
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import uuid
from training_helper import get_preprocessing, get_training_augmentation, get_validation_augmentation
ssl._create_default_https_context = ssl._create_unverified_context
import json
import wandb
import sys


class RecordingWeightedSampler(WeightedRandomSampler):
    """WeightedRandomSampler that records the drawn indices for post-epoch analysis."""

    def __iter__(self):
        self.last_indices = list(super().__iter__())
        return iter(self.last_indices)


class FocalLossIgnoreBackground(torch.nn.Module):
    """Focal loss that ignores pixels where all target channels are zero (background).

    Used when ignore_background=True (no background channel in the model).
    All-zero target pixels would otherwise push every foreground logit down on
    the majority of the image, swamping the sparse foreground signal.
    Expects y_pred as logits — FocalLoss(multilabel) applies logsigmoid internally.
    """
    __name__ = 'Focal_loss'

    def __init__(self):
        super().__init__()
        self._focal = smp.losses.FocalLoss(mode='multilabel')

    def forward(self, y_pred, y_true):
        # y_true: (B, C, H, W) one-hot; background pixels have all channels = 0
        fg_mask = y_true.sum(dim=1) > 0  # (B, H, W)
        if not fg_mask.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        # Select only foreground pixels and reshape to (N_fg, C, 1, 1) for SMP loss
        y_pred_fg = y_pred.permute(0, 2, 3, 1)[fg_mask].unsqueeze(-1).unsqueeze(-1)
        y_true_fg = y_true.permute(0, 2, 3, 1)[fg_mask].unsqueeze(-1).unsqueeze(-1)
        return self._focal(y_pred_fg, y_true_fg)


class IoUIgnoreBackground(torch.nn.Module):
    """IoU metric that ignores background pixels (where all target channels are zero).

    Background pixels can create false positives when the softmax output exceeds the
    threshold for any class. Zeroing out predictions on background pixels eliminates
    these spurious false positives without affecting TP or FN counts.
    Expects y_pred as logits — applies softmax internally.
    """
    __name__ = 'iou_score'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        fg_mask = (y_true.sum(dim=1, keepdim=True) > 0).float()  # (B, 1, H, W)
        # Zero out predictions on background pixels so they cannot become false positives
        return self._iou(y_pred * fg_mask, y_true)


class MultiClassFocalLoss(torch.nn.Module):
    """Focal loss in multiclass mode. Background is trained as an explicit class
    rather than being masked out of the loss — focal's (1-p)^gamma term
    down-weights easy (typically background) pixels, so they stop drowning out
    foreground gradient without the model losing its ability to predict background.
    Converts one-hot targets (B, C, H, W) -> class-index maps (B, H, W).
    Expects y_pred as logits — FocalLoss(multiclass) applies log_softmax internally.
    """
    __name__ = 'Focal_loss'

    def __init__(self):
        super().__init__()
        self._focal = smp.losses.FocalLoss(mode='multiclass')

    def forward(self, y_pred, y_true):
        y_true_idx = y_true.argmax(dim=1).long()
        return self._focal(y_pred, y_true_idx)


class IoUForegroundOnly(torch.nn.Module):
    """IoU over foreground channels only (excludes channel 0 = background).
    Keeps the reported metric comparable to IoUIgnoreBackground while the model
    is trained on all classes including background.
    Expects y_pred as logits — applies softmax internally.
    """
    __name__ = 'iou_score'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        return self._iou(y_pred[:, 1:], y_true[:, 1:])


class IoUBackgroundOnly(torch.nn.Module):
    """IoU over the background channel only (channel 0). Reported alongside
    the foreground IoU so the model's background-vs-foreground behavior is
    visible without contaminating the main model-selection metric.
    Expects y_pred as logits — applies softmax internally.
    """
    __name__ = 'iou_background'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        return self._iou(y_pred[:, :1], y_true[:, :1])


class Trainer():
    """Trainer class for training and testing segmentation model
    Used for manual training on GPU server, for model testing/evaluation and
    for predicting images
    """

    def __init__(self, 
                 encoder='efficientnet-b3', 
                 encoder_weights='imagenet',
                 optimizer_name='Adam',
                 train_split=0.8,
                 epochs=1000,
                 seed=10,
                 load_config=True,
                 device='cuda', 
                 size=1024, 
                 pred=False, 
                 cross_validation=False,
                 n_folds=0,
                 save_val_uncertainty=False,
                 config_file="None",
                 cross_val_exp_series="none",
                 memory_optimized=False,
                 batch_size=1,
                 accumulation_steps=4,
                 data_leakage=False,
                 ignore_background=False,
                 use_weighted_sampler=False,
                 num_workers=4
                 ):

        self.size = size
        self.pred = pred
        self.seed = seed
        self.dataset = None
        self.research_site = None
        self.save_val_uncertainty = save_val_uncertainty
        self.ignore_background = ignore_background
        self.use_weighted_sampler = use_weighted_sampler
        self.config_file = config_file
        self.device = device
        self.epochs = epochs
        self.train_split = train_split
        self.cross_validation = cross_validation
        self.cross_val_run = 0
        self.n_folds = n_folds
        self.cross_val_exp_series = cross_val_exp_series
        self.memory_optimized = memory_optimized
        self.accumulation_steps = accumulation_steps
        self.data_leakage = data_leakage
        self.num_workers = 0 if self.memory_optimized else num_workers

        
        if load_config:
            self.ontology_file_name = None
            self.load_config()
            self.load_ontology()
            if self.ignore_background:
                self.class_values = [d["value"] for d in self.ontology["ontology"].values() if d["value"] != 0]
                self.labels = list(self.ontology["ontology"].keys())[1:]
            else:
                self.class_values = [d["value"] for d in self.ontology["ontology"].values()]
                self.labels = list(self.ontology["ontology"].keys())

        # model settings
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        # activation=None so the model returns logits. smp's FocalLoss
        # (multiclass/multilabel) applies log_softmax/logsigmoid internally;
        # pre-applying softmax would double-activate and flatten gradients.
        # Softmax is applied explicitly where probabilities are needed
        # (IoU metric wrappers and inference helpers).
        self.activation = None
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.lr = 1e-4
        self.uncertainy_runs = 10
        self.weight_decay = 1e-4
        self.lr_scheduler = "ReduceLROnPlateau"

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.preprocessing = get_preprocessing(self.preprocessing_fn)

        self.check_gpu()



    def check_gpu(self):
        # --- GPU/CUDA Diagnostics ---
        print("=== CUDA Diagnostics ===")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")

        # Check what GPUs are physically visible
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                    '--format=csv,noheader'], capture_output=True, text=True)
            print(f"nvidia-smi output:\n{result.stdout.strip()}")
            print(f"nvidia-smi stderr: {result.stderr.strip()}")
        except FileNotFoundError:
            print("nvidia-smi not found!")

        print("=" * 40)


        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"pynvml sees {count} GPUs")
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                print(f"  GPU {i}: {pynvml.nvmlDeviceGetName(h)}")
            except Exception as e:
                print(f"  GPU {i}: ERROR — {e}")

        # GPU Memory Debug
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n=== GPU Memory Before Loading Model ===")
            print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
            print("=" * 40 + "\n")

        

    def load_ontology(self):
        """set ontologies, should be moved to separate file!!!
        """

        if self.ontology_file_name is None:
            self.ontology_file_name = "ontology_{0}.json".format(self.ontology_name)
        
        with open(self.ontology_file_name) as f:
            self.ontology = json.load(f)
            print(self.ontology)
            
            
    def set_seed(self, seed):
        """set seed for reproducibility

        Args:
            seed (int): seed value
        """
        self.seed = seed

    def set_encoder(self, encoder):
        """set encoder for model

        Args:
            encoder (str): encoder name
        """
        self.encoder = encoder

    def load_dataset_info(self):
        self.dataset = self.yaml_file["dataset"]
        dataset_info_path = os.path.join(self.yaml_file["img_dir"], self.dataset, "dataset_info.json")
        with open(dataset_info_path) as f:
            self.dataset_info = json.load(f)
            self.ontology_name = self.dataset_info["ontology_file"].split(".json")[0].split("ontology_")[-1]
            self.ontology_file_name = self.dataset_info["ontology_file"]
            self.wandb_project = self.dataset_info["wandb_project"]
            self.wandb_logged_artifact = self.dataset_info["logged_artifact"]
            self.wandb_qualified_name = self.dataset_info["qualified_name"]

            if self.ontology_name == "atto":
                self.research_site = self.wandb_logged_artifact.split("_")[1]
                print("Research site: ", self.research_site)

    def load_config(self):
        """ load server config by default, if not available load local config
        """

        if self.config_file == "None":
            self.config_file = "server_config.yaml"
                

        with open(self.config_file) as f:
            self.yaml_file = yaml.safe_load(f)
            self.best_model_path = self.yaml_file["best_model_path"]
            self.use_gpu = self.yaml_file["use_gpu"]
            if not self.use_gpu:
                self.device = 'cpu'
                print("Local testing? \n Using CPU! \n GPU NOT ACTIVATED!!!")
            print(self.best_model_path)

        if "dataset" in self.yaml_file:
            self.load_dataset_info()

        else:
            self.ontology_name = self.yaml_file["ontology"]
            self.ontology_file_name = "ontology_{0}.json".format(self.ontology_name)
            self.size = self.yaml_file["size"]
            self.wandb_project = self.yaml_file["train_img"]

        # load local config if path does not exist
        if not Path(self.yaml_file["img_dir"]).exists():
            print("IMG Path does not exist\n Fall back to local config!")
            with open("local_config.yaml") as f:
                self.yaml_file = yaml.safe_load(f)
                self.best_model_path = self.yaml_file["best_model_path"]


        if self.dataset is not None:
            # Create NEW PATHS SERVER
            self.img_dir = Path(os.path.join(self.yaml_file["img_dir"], self.yaml_file["dataset"], "partial_images"))
            self.mask_dir = Path(os.path.join(self.yaml_file["mask_dir"], self.yaml_file["dataset"], "partial_masks"))

        else:
            # Create OLD PATHS SERVER
            self.img_dir = Path(os.path.join(self.yaml_file["img_dir"], str(self.yaml_file["train_img"])))
            self.mask_dir = Path(os.path.join(self.yaml_file["mask_dir"], str(self.yaml_file["train_mask"])))

        self.test_dir = Path(os.path.join(self.yaml_file["test_dir"]))



    def get_git_commit_hash(self):
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:7]  # Short hash
        except subprocess.CalledProcessError:
            return "unknown"
        

    def setup_pred_logging(self):
           # Get git commit hash
        git_hash = self.get_git_commit_hash()

        wandb.init(
            project=self.wandb_project,
            settings=wandb.Settings(
                    _cuda_device_ids=[0]  # index 0 within CUDA_VISIBLE_DEVICES, i.e. your actual GPU
                ),
            name=self.exp_dir.stem,
            job_type='ml-prediction',
            config={
                
                # Model architecture
                "model": "Unet",
                "encoder": self.encoder,
                "encoder_weights": "imagenet",
                "classes": len(self.class_values),
                
                
                # Random seeds
                "seed": self.seed,
                "torch_seed": self.seed,
                "numpy_seed": self.seed,
                
                # Environment
                "git_commit": git_hash,
                "cuda_version": torch.version.cuda,
                "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "num_workers": self.num_workers
            }
        )

        # Save pip freeze snapshot
        wandb.save("requirements.txt")  # if you generate it before training

    def setup_logging(self):
        """setup logging for training and validation
        """
        # Get git commit hash
        git_hash = self.get_git_commit_hash()

        wandb.init(
            project=self.wandb_project,
            name=self.exp_dir.stem,
            job_type='ml-optimizition',
            config={
                # Train and Val images
                "len_unique_train_images" : self.len_unique_train_imgs,
                "len_unique_val_images" : self.len_unique_val_imgs,
                "unique_train_images" : self.unique_train_imgs,
                "unique_val_images" : self.unique_val_imgs,
                # Hyperparameters
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "optimizer": self.optimizer_name,
                "activation": self.activation,
                "weight_decay": self.weight_decay,
                "scheduler": self.lr_scheduler,
                "accumulation_steps": self.accumulation_steps,
                
                # Model architecture
                "model": "Unet",
                "encoder": self.encoder,
                "encoder_weights": "imagenet",
                "classes": len(self.class_values),
                
                # Dataset info (link to your dataset artifact)
                "dataset_name": self.dataset,
                "dataset_version": self.dataset_info["version"],
                
                # Random seeds
                "seed": self.seed,
                "torch_seed": self.seed,
                "numpy_seed": self.seed,
                
                # Data augmentation (save transform configs)
                # "augmentations": str(train_transforms),  # or serialize to dict
                
                # Loss function
                "loss_function": self.loss.__name__,
                # "class_weights": None,
                
                # Environment
                "git_commit": git_hash,
                "cuda_version": torch.version.cuda,
                "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            }
        )

        dataset_artifact = wandb.use_artifact(self.wandb_qualified_name, type='dataset')
        # Save pip freeze snapshot
        wandb.save("requirements.txt")  # if you generate it before training


    def set_non_unique_paths(self, train_split):
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

        self.len_unique_train_imgs = 0
        self.len_unique_val_imgs = 0

        self.unique_train_imgs = []
        self.unique_val_imgs = []


    def split_train_val(self, shuffled_original):

            val_fold_size = len(shuffled_original) // self.n_folds
            val_start = self.cross_val_run * val_fold_size
            val_end = val_start + val_fold_size if self.cross_val_run < self.n_folds - 1 else len(shuffled_original)
            
            unique_val_imgs = shuffled_original[val_start:val_end]
            unique_train_imgs = shuffled_original[:val_start] + shuffled_original[val_end:]
            print("Val start: ", val_start)
            print("Val end: ", val_end)
            print("Train images: ", len(unique_train_imgs))
            print("Val images: ", len(unique_val_imgs))

            return unique_train_imgs, unique_val_imgs


    def get_mixed_train_val_imgs(self, sorted_unique_imgs):

        campina_imgs = [x for x in sorted_unique_imgs if x.split("_")[1] == "C"]
        terra_firme_imgs = [x for x in sorted_unique_imgs if x.split("_")[1] == "TF"]

        np.random.shuffle(campina_imgs)
        np.random.shuffle(terra_firme_imgs)

        tf_shuffled_unique_train_imgs, tf_shuffled_unique_val_imgs = self.split_train_val(terra_firme_imgs)
        c_shuffled_unique_train_imgs, c_shuffled_unique_val_imgs = self.split_train_val(campina_imgs)

        unique_train_imgs = tf_shuffled_unique_train_imgs + c_shuffled_unique_train_imgs
        unique_val_imgs = tf_shuffled_unique_val_imgs + c_shuffled_unique_val_imgs

        return unique_train_imgs, unique_val_imgs

    def get_mixed_snow_train_val_imgs(self, sorted_unique_imgs):

        snow_imgs = [x for x in sorted_unique_imgs if "september" in x]
        non_snow_imgs = [x for x in sorted_unique_imgs if "september" not in x]

        np.random.shuffle(snow_imgs)
        np.random.shuffle(non_snow_imgs)

        non_snow_shuffled_unique_train_imgs, non_snow_shuffled_unique_val_imgs = self.split_train_val(non_snow_imgs)
        snow_shuffled_unique_train_imgs, snow_shuffled_unique_val_imgs = self.split_train_val(snow_imgs)

        unique_train_imgs = non_snow_shuffled_unique_train_imgs + snow_shuffled_unique_train_imgs
        unique_val_imgs = non_snow_shuffled_unique_val_imgs + snow_shuffled_unique_val_imgs

        return unique_train_imgs, unique_val_imgs

    def get_train_val_imgs(self, sorted_unique_imgs, train_split):
        
        shuffled_original = sorted_unique_imgs.copy()
        np.random.shuffle(shuffled_original)


        if self.cross_validation:
            unique_train_imgs, unique_val_imgs = self.split_train_val(shuffled_original)
        else:
            train_len_ = int(len(sorted_unique_imgs) * train_split)
            unique_train_imgs = shuffled_original[:train_len_]
            unique_val_imgs = shuffled_original[train_len_:]

        return unique_train_imgs, unique_val_imgs

    def set_unique_paths(self, train_split, sorted_unique_imgs):

        if self.research_site == "mixed":
            unique_train_imgs, unique_val_imgs = self.get_mixed_train_val_imgs(sorted_unique_imgs, train_split)
        
        
        else:
            unique_train_imgs, unique_val_imgs = self.get_train_val_imgs(sorted_unique_imgs, train_split)

        self.unique_train_imgs = unique_train_imgs
        self.unique_val_imgs = unique_val_imgs

        self.len_unique_train_imgs = len(unique_train_imgs)
        self.len_unique_val_imgs = len(unique_val_imgs)

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



    def set_paths(self, cross_val_run=0, train=False, test=False, pred=False):
        """Set train, valid, test paths for images and masks.
        """

        self.cross_val_run = cross_val_run

        if train:
            # check if exp_dir exists and create new one
            self.exp_dir = Path(os.path.join(self.yaml_file["exp_dir"], "exp_{0}_{1}_s{2}".format(self.wandb_project, self.encoder, self.seed)))
            if self.cross_validation:
                self.exp_dir = Path(str(self.exp_dir) + "_cv{0}_".format(self.cross_val_run))
                # self.cross_val_exp_series = str(uuid.uuid4())[:6]
                exp_dir = Path(str(self.exp_dir) + str(self.cross_val_exp_series))
            else:
                exp_dir = Path(str(self.exp_dir) + "_" + str(uuid.uuid4())[:6])
                while exp_dir.exists():
                    exp_dir = Path(str(self.exp_dir) + "_" + str(uuid.uuid4())[:6])

            self.exp_dir = exp_dir
            os.makedirs(self.exp_dir)

            self.model_save_path = os.path.join(self.exp_dir, 'best_model.pth')

        if test:
            self.exp_dirs = [x for x in os.listdir(Path(self.yaml_file["exp_dir"])) if encoder in x]
            self.sorted_exp_dirs = sorted(self.exp_dirs, key=lambda x: int(x.split("_")[-1]))
            self.exp_dir = Path(os.path.join(Path(self.yaml_file["exp_dir"]), self.sorted_exp_dirs[-1]))
            print(self.exp_dir)
        if pred:
            self.exp_dir = Path(os.path.join(self.yaml_file["exp_dir"], "exp_{0}_{1}".format(self.encoder, self.seed)))
            self.model_save_path = self.best_model_path
            print("Best model path: ", self.best_model_path)
            print("Predicting images from: ", self.test_dir)

        pdf_path = os.path.join(self.exp_dir, "results.pdf")
        count = 1
        while os.path.exists(pdf_path):
            pdf_path = os.path.join(self.exp_dir, "results_{0}.pdf".format(count))
            count += 1
            
        self.pdf_path = pdf_path


        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.ids = sorted(os.listdir(self.img_dir))

        # get unique image names
        original_imgs = [x.split("_part")[0] for x in self.ids]
        sorted_unique_imgs = sorted(np.unique(original_imgs))
        print(sorted_unique_imgs)
        if len(sorted_unique_imgs) == 0:
            print("Error: No images found!")
            return None

        num_unique = 10 # 10 for training; 1000 just for data leakage test

        if len(sorted_unique_imgs) < num_unique or self.data_leakage:
            print("Only {0} image found! Using non-unique images. Data leakage might be a problem...".format(len(sorted_unique_imgs)))
            self.set_non_unique_paths(self.train_split)

        elif len(sorted_unique_imgs) >= num_unique:
            print("Multiple images found! Using unique images to tackle data leakage")
            self.set_unique_paths(self.train_split, sorted_unique_imgs)


        self.check_class_distribution()

    def check_class_distribution(self):

        print("\n=== Class Distribution Analysis ===")
        train_class_counts = np.zeros(len(self.class_values))
        val_class_counts = np.zeros(len(self.class_values))
        train_patch_counts = []

        # First pass: count pixels per class globally and per tile
        for mask_path in self.y_train:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            patch_counts = np.zeros(len(self.class_values))
            for idx, class_val in enumerate(self.class_values):
                count = np.sum(mask == class_val)
                train_class_counts[idx] += count
                patch_counts[idx] = count
            train_patch_counts.append(patch_counts)

        self.train_patch_counts = np.array(train_patch_counts)  # (N_tiles, N_classes)

        # Per-tile sampling weight = expected inverse-global-frequency over the
        # tile's pixels. Tiles containing globally rare classes are boosted;
        # tiles dominated by globally common classes are damped. Unlike the
        # prior local-rarity formula, a tile that is 100% of a globally rare
        # class is correctly oversampled.
        global_inv_freq = 1.0 / np.maximum(train_class_counts, 1.0)
        tile_totals = self.train_patch_counts.sum(axis=1)
        safe_totals = np.where(tile_totals > 0, tile_totals, 1)
        tile_fractions = self.train_patch_counts / safe_totals[:, None]
        tile_weights = tile_fractions @ global_inv_freq
        tile_weights = np.where(tile_totals > 0, tile_weights, 0.0)
        self.sample_weights = tile_weights.tolist()
        
        # Count validation set
        for mask_path in self.y_valid:
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


    def create_dataloaders(self, augmentations=True):
        """creates training and validation Datasets (using smp_dataset.py) and Dataloaders
        if augmentations is True, training and validation augmentations are used

        Args:
            augmentations (bool, optional): use/dont use augmentations. Defaults to True.
        """
        if augmentations:
            training_augmentation = get_training_augmentation(min_height=1024, min_width=1024)
            validation_augmentation = get_validation_augmentation(min_height=1024, min_width=1024)
        else:
            training_augmentation = None
            validation_augmentation = None

        self.train_dataset = Dataset(
            self.x_train, 
            self.y_train,
            class_values=self.class_values,
            augmentation=training_augmentation, 
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        self.valid_dataset = Dataset(
            self.x_valid, 
            self.y_valid,
            class_values=self.class_values,
            augmentation=validation_augmentation, 
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        loader_kwargs = dict(pin_memory=True, prefetch_factor=4, persistent_workers=True) if self.num_workers > 0 else {}

        if self.use_weighted_sampler:
            sampler = RecordingWeightedSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )
        else:
            sampler = None

        self.sampler = sampler
        
        self.train_loader = DataLoader(self.train_dataset,
                                        batch_size=self.batch_size,
                                        sampler=sampler,
                                        num_workers=self.num_workers,
                                        **loader_kwargs,
                                       )

        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers,
                                       **loader_kwargs,
                                       )


    def update_dataloaders(self, x_train):
        """update training dataloader with new images

        Args:
            x_train (list): list of training image paths
            x_valid (list): list of validation image paths
        """
        y_train = [x.replace("JPG", "png").replace("img", "mask") for x in x_train]
        # y_valid = [x.replace("JPG", "png").replace("img", "mask") for x in x_valid]

        self.x_train += x_train
        self.y_train += y_train

        self.train_dataset = Dataset(
            self.x_train, 
            self.y_train,
            class_values=self.class_values,
            augmentation=get_training_augmentation(min_height=1024, min_width=1024), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)



    def prepare_model(self):
        """Prepare model for loading (training or testing/predicting).
        Set loss, metrics, optimizer and training / validation runners.
        """

        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name=self.encoder, 
            encoder_weights=self.encoder_weights, 
            classes=len(self.class_values), 
            activation=self.activation,
        )

        self.model = self.model.to(self.device)

        #  self.dice_loss = smp.losses.DiceLoss(mode='multilabel', ignore_index=[0])
        # self.dice_loss.__name__ = 'Dice_loss'
        # self.loss = self.dice_loss

        if self.memory_optimized and 'mit_' in self.encoder:
            from torch.utils.checkpoint import checkpoint
            
            encoder = self.model.encoder
            
            # Wrap each block's forward to use checkpointing
            for block_name in ['block1', 'block2', 'block3', 'block4']:
                block = getattr(encoder, block_name, None)
                if block is not None:
                    original_forward = block.forward
                    
                    def make_checkpointed_forward(orig_fwd):
                        def checkpointed_forward(x):
                            # Checkpoint each sub-block in the Sequential
                            for sub_block in orig_fwd.__self__:
                                x = checkpoint(sub_block, x, use_reentrant=False)
                            return x
                        return checkpointed_forward
                    
                    block.forward = make_checkpointed_forward(original_forward)
            
            print("Gradient checkpointing enabled for MiT encoder")

        
        if self.ignore_background:
            self.focal_loss = FocalLossIgnoreBackground()
            self.metrics = [IoUIgnoreBackground(threshold=0.5)]
        else:
            self.focal_loss = MultiClassFocalLoss()
            self.metrics = [IoUForegroundOnly(threshold=0.5), IoUBackgroundOnly(threshold=0.5)]
        self.loss = self.focal_loss


        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            ])
        elif self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD([ 
            dict(params=self.model.parameters(), lr=self.lr),
            ])
        else:
            self.optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            ])

        if self.lr_scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',           # Maximize validation IoU
                factor=0.5,           # Cut LR in half
                patience=20,          # Wait 20 epochs before reducing
                verbose=True
            )

        # create epoch runners 
        # it is a simple loop of iterating over dataloader`s samples
        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=self.loss, 
            metrics=self.metrics, 
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        self.valid_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=self.loss, 
            metrics=self.metrics, 
            device=self.device,
            verbose=True,
        )

        if self.pred:
            self.setup_pred_logging()
        else:
            self.setup_logging()


    def save_model(self, epoch, update_latest=False):
        """save model to model_save_path (experiment folder + best_model.pth)
        train/valid dataloaders seem to need a lot of memory and are not saved in the model
        """
        old_ontology_dict = {class_name: data["color"] for class_name, data in self.ontology["ontology"].items()}
        ckpt_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'train_dl': self.train_loader,
            # 'valid_dl': self.valid_loader,
            'epoch': epoch,
            'ontology' : old_ontology_dict,
            'encoder': self.encoder
        }
        if not update_latest:
            torch.save(ckpt_dict, self.model_save_path)
        else:
            save_path = os.path.join(self.exp_dir, 'latest_model.pth')
            torch.save(ckpt_dict, save_path)
        
        print("Model updated!")


    def get_per_image_ious(self):
            # Track per-image performance
            print("\n=== Per-Image Validation IoU ===")
            
            image_ious = []
            for i, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    pred = self.model(x)
                    iou = self.metrics[0](pred, y).item()
                    image_name = Path(self.x_valid[i]).name
                    image_ious.append((image_name, iou))
            
            return image_ious
            

    def run_train_loop(self):
        max_score = 0
        self.image_iou_df = pd.DataFrame({'image': [Path(x).stem for x in self.x_valid]})
        self.image_iou_csv_path = os.path.join(self.exp_dir, "per_image_iou_tracking.csv")

        for epoch in range(0, self.epochs):

            print("cuda available: ", torch.cuda.is_available())
            print(f"[GPU] Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB | "
            f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
            
            print('\n Seed {0} | Epoch: {1}/{2}'.format(self.seed, epoch, self.epochs))
            self.train_logs = self.train_epoch.run(self.train_loader)
            self.valid_logs = self.valid_epoch.run(self.valid_loader)

            # image_ious = self.get_per_image_ious()
            
            # do something (save model, change lr, etc.)
            if max_score < self.valid_logs['iou_score']:
                print('New top score: {0} > {1} '.format(self.valid_logs['iou_score'], max_score))
                max_score = self.valid_logs['iou_score']
        
                self.save_model(epoch=epoch)
                self.save_loss_plots()

                if self.save_val_uncertainty:
                    self.save_uncertainty_images(epoch)

            self.save_confusion_matrix(epoch=epoch)

            self.train_log_df.loc[epoch] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            self.valid_log_df.loc[epoch] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score']]

            self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
            self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))


            # iou_values = [iou for _, iou in image_ious]
            # image_ious_sorted = sorted(image_ious, key=lambda x: x[1])

            # pd.DataFrame(image_ious, columns=['image', 'iou']).to_csv(
            #     os.path.join(self.exp_dir, f"image_ious_epoch_{epoch}.csv"),
            #     index=False
            #     )

            wandb_log = {
                "epoch": epoch,
                "train/loss": self.train_logs[self.loss.__name__],
                "train/iou": self.train_logs["iou_score"],
                "val/loss": self.valid_logs[self.loss.__name__],
                "val/iou": self.valid_logs["iou_score"],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                # Per-image IoU statistics
                # "val/iou_min": min(iou_values),
                # "val/iou_max": max(iou_values),
                # "val/iou_std": np.std(iou_values),
                # "val/worst_image": f"{image_ious_sorted[0][0]}: {image_ious_sorted[0][1]:.3f}",
                # "val/best_image": f"{image_ious_sorted[-1][0]}: {image_ious_sorted[-1][1]:.3f}",
            }
            if 'iou_background' in self.valid_logs:
                wandb_log["train/iou_background"] = self.train_logs['iou_background']
                wandb_log["val/iou_background"] = self.valid_logs['iou_background']
            wandb.log(wandb_log)

            self.scheduler.step(self.valid_logs['iou_score'])



    def run_train_loop_with_accumulation(self, early_stopping_patience=0):
        """Run training loop with gradient accumulation.

        Args:
            early_stopping_patience (int): stop if val IoU does not improve for this many
                epochs. 0 (default) disables early stopping.
        """
        max_score = 0
        epochs_without_improvement = 0

        print("device: ", self.device)
        for epoch in range(self.epochs):
            print(f'\n Seed {self.seed} | Epoch: {epoch}/{self.epochs}')

            # === TRAINING ===
            self.model.train()
            train_loss_sum = 0
            train_metric_sums = {m.__name__: 0.0 for m in self.metrics}
            self.optimizer.zero_grad()
            pbar = tqdm(self.train_loader, desc='train')

            print("cuda available: ", torch.cuda.is_available())
            print(f"[GPU] Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB | "
            f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.loss(pred, y)
                (loss / self.accumulation_steps).backward()

                train_loss_sum += loss.item()
                with torch.no_grad():
                    for m in self.metrics:
                        train_metric_sums[m.__name__] += m(pred.detach(), y).item()
                postfix = {'loss': train_loss_sum/(batch_idx+1)}
                for name, total in train_metric_sums.items():
                    postfix[name] = total / (batch_idx + 1)
                postfix['gpu_mem'] = f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB"
                pbar.set_postfix(**postfix)

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # === VALIDATION ===
            self.model.eval()
            val_loss_sum = 0
            val_metric_sums = {m.__name__: 0.0 for m in self.metrics}

            with torch.no_grad():
                for x, y in tqdm(self.valid_loader, desc='valid'):
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = self.loss(pred, y)
                    val_loss_sum += loss.item()
                    for m in self.metrics:
                        val_metric_sums[m.__name__] += m(pred, y).item()

            # === LOGGING ===
            self.train_logs = {
                self.loss.__name__: train_loss_sum / len(self.train_loader),
                **{name: total / len(self.train_loader) for name, total in train_metric_sums.items()}
            }
            self.valid_logs = {
                self.loss.__name__: val_loss_sum / len(self.valid_loader),
                **{name: total / len(self.valid_loader) for name, total in val_metric_sums.items()}
            }

            if max_score < self.valid_logs['iou_score']:
                print(f'New top score: {self.valid_logs["iou_score"]:.4f} > {max_score:.4f}')
                max_score = self.valid_logs['iou_score']
                epochs_without_improvement = 0
                self.save_model(epoch=epoch)
                self.save_loss_plots()
                if self.save_val_uncertainty:
                    self.save_uncertainty_images(epoch)
            else:
                epochs_without_improvement += 1

            self.save_confusion_matrix(epoch=epoch)
            self.train_log_df.loc[epoch] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            self.valid_log_df.loc[epoch] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score']]
            self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
            self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))

            self.scheduler.step(self.valid_logs['iou_score'])

            sampled_counts = self.train_patch_counts[self.sampler.last_indices].sum(axis=0)
            sampled_total = sampled_counts.sum()
            sampled_dist = {
                f"sampled_dist/{label}": sampled_counts[i] / sampled_total if sampled_total > 0 else 0
                for i, label in enumerate(self.labels)
            }

            wandb_log = {
                "epoch": epoch,
                "train/loss": self.train_logs[self.loss.__name__],
                "train/iou": self.train_logs["iou_score"],
                "val/loss": self.valid_logs[self.loss.__name__],
                "val/iou": self.valid_logs["iou_score"],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                **sampled_dist,
            }
            if 'iou_background' in self.valid_logs:
                wandb_log["train/iou_background"] = self.train_logs['iou_background']
                wandb_log["val/iou_background"] = self.valid_logs['iou_background']
            wandb.log(wandb_log)

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping: no improvement for {early_stopping_patience} epochs.')
                break

    def train_model(self, al_step=25, early_stopping_patience=0):

        torch.cuda.empty_cache()

        print("Training on {0} images".format(len(self.ids)))
        if self.cross_validation:
            print("Cross validation is enabled!")
            print("Cross val run: ", self.cross_val_run)

        self.train_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])
        self.valid_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])


        if self.memory_optimized:
            self.run_optimized_train_loop(al_step=al_step)
        else:
            if self.accumulation_steps == 0:
                self.run_train_loop()
            else:
                self.run_train_loop_with_accumulation(early_stopping_patience=early_stopping_patience)
        self.save_loss_plots()




        self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
        self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))


    def save_confusion_matrix(self, epoch):
        cf_matrix = np.zeros((len(self.class_values), len(self.class_values)))
        # Map raw pixel values to matrix indices (e.g. class value 9 → index 8 when background excluded)
        val_to_idx = {v: i for i, v in enumerate(self.class_values)}

        for n in range(len(self.valid_dataset)):
            image, gt_mask = self.valid_dataset[n]
            pred = self.calculate_prediction(image)
            gt_mask_2d, _ = self.get_2d_image(gt_mask)

            y_pred_flattened = pred.flatten()
            y_true_flattened = gt_mask_2d.flatten()

            # Use only foreground labels present in y_true for this image.
            # Background-only patches (no class_values in y_true) are skipped.
            # y_pred values outside these labels are ignored by sklearn (treated as misses).
            true_labels = sorted(
                {v for v in np.unique(y_true_flattened) if v in val_to_idx}
            )
            present_labels = sorted(
                {v for v in np.unique(np.concatenate([y_true_flattened, y_pred_flattened]))
                 if v in val_to_idx}
            )
            if not true_labels:
                continue

            # DEBUG
            # print(f"[CFM DEBUG] image {n}")
            # print(f"  y_true unique: {np.unique(y_true_flattened)}")
            # print(f"  y_pred unique: {np.unique(y_pred_flattened)}")
            # print(f"  true_labels: {true_labels}")
            # print(f"  present_labels (true+pred): {present_labels}")
            # print(f"  class_values: {self.class_values}")
            # print(f"  val_to_idx: {val_to_idx}")

            cf_m = confusion_matrix(y_true_flattened, y_pred_flattened, labels=true_labels)
            for i, u_val in enumerate(true_labels):
                for j, k_val in enumerate(true_labels):
                    cf_matrix[val_to_idx[u_val]][val_to_idx[k_val]] += cf_m[i][j]


        cf_matrix_normed = normalize(cf_matrix, axis=1, norm='l1')
        disp = ConfusionMatrixDisplay(cf_matrix_normed, display_labels=self.labels)

        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        plt.colorbar(disp.im_, boundaries=np.linspace(0, 1, 11))
        plt.savefig(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.png".format(epoch)))

        cfm_df = pd.DataFrame(columns=[self.labels], index=self.labels, data=cf_matrix_normed)
        cfm_df.to_csv(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.csv".format(epoch)))

        # Raw (unnormalized) matrix preserves row/column totals so per-class IoU
        # (including background) can be recomputed offline: IoU_c = CM[c,c] /
        # (sum(CM[c,:]) + sum(CM[:,c]) - CM[c,c])
        cfm_raw_df = pd.DataFrame(columns=[self.labels], index=self.labels, data=cf_matrix)
        cfm_raw_df.to_csv(os.path.join(self.exp_dir, "confusion_matrix_raw_ep{0}.csv".format(epoch)))



    def save_loss_plots(self):
        """save plots for training and validation loss
        """
        self.pdf = PdfPages(self.pdf_path)

        title = "Model: {0}, Seed: {1}".format(self.encoder, self.seed)
        fig, ax = plt.subplots(figsize=(16, 5), ncols=2)
        fig.suptitle(title)

        ax[0].plot(self.train_log_df.index, self.train_log_df[self.loss.__name__], label='train {0} loss'.format(self.loss.__name__))
        ax[0].plot(self.valid_log_df.index, self.valid_log_df[self.loss.__name__], label='valid {0} loss'.format(self.loss.__name__))
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(self.train_log_df.index, self.train_log_df['iou_score'], label='train iou score')
        ax[1].plot(self.valid_log_df.index, self.valid_log_df['iou_score'], label='valid iou score')
        # ax[1].set_ylim(0.6, 1.0)
        ax[1].grid()
        ax[1].legend()

        plt.show()

        self.pdf.savefig(fig)
        self.pdf.close()
    
    def get_2d_image(self, mask):
        """convert mask with dimension N_CLASSESxHEIGHTxWIDTH to 2d mask (HEIGHTxWIDTH)

        Args:
            mask np.array: prediction mask with one hot encoding (N_CLASSESxHEIGHTxWIDTH)

        Returns:
            mask: prediction mask with class encoding [0...N_CLASSES] (HEIGHTxWIDTH)
        """
        mask = mask.transpose(1, 2, 0)
        out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

        # for v in self.class_values:
            # out[mask[:,:,v] == 1] = v

        for channel_idx, class_val in enumerate(self.class_values): # consider removed background
            out[mask[:, :, channel_idx] == 1] = class_val
        mask = out
        unique_values = np.unique(mask)

        return mask, unique_values
        
    def save_images_to_pdf(self, title=None, **images):
        """
        create plots for training and validation loss,
        visualize images in a row: input image, ground truth and prediction
        """
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.patches as mpatches

        n = len(images)

        # colors_hex = ["#000000","#1cffbb", "#00bcff","#0059ff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
        colors_hex = []

        for key in self.ontology["ontology"].keys():
            colors_hex.append(self.ontology["ontology"][key]["color"])

        col = ListedColormap(colors_hex)
        bounds = np.arange(len(colors_hex)+1)
        norm = BoundaryNorm(bounds, col.N)  

        fig = plt.figure(figsize=(16, 10))

        if title:
            fig.suptitle(title)

        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())

            if not name == 'image':
                image, unique_values = self.get_2d_image(image)

                plt.imshow(image, cmap=col, interpolation='nearest', norm=norm)
                # plot only unique class values in legend
                if self.ignore_background:
                    # labels has no background entry, so pixel value i maps to labels[i-1]; skip background pixels (i=0)
                    patches = [mpatches.Patch(color=colors_hex[i], label=self.labels[i - 1]) for i in unique_values if i > 0]
                else:
                    # labels includes background at index 0, so pixel value i maps directly to labels[i]
                    patches = [mpatches.Patch(color=colors_hex[i], label=self.labels[i]) for i in unique_values]

                plt.axis('off')
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0. )   
            else:
                plt.imshow(image)
        # plt.show()
        self.pdf.savefig(fig)
        plt.close()

    def print_summary(self, checkpoint):
        """Print summary of model and training results (before loading)

        Args:
            checkpoint (dictionary): checkpoint dictionary of ml-model
        """

        print("Summary: ")
        print("Epoch: ", checkpoint['epoch'])
        print("Ontology: ", checkpoint['ontology'])

    def load_model(self):
        """load checkpoint from model_save_path
        after loading model (and if there is no best_model_path)
        set exp_dir to parent of model_save_path
        """

        print("Loading checkpoint from: ")
        if not self.pred:
            print(self.model_save_path)
            # checkpoint = torch.load(self.model_save_path)
            checkpoint = torch.load(self.model_save_path, map_location=self.device)
        elif self.pred:
            print(self.best_model_path)
            # checkpoint = torch.load(self.best_model_path)
            checkpoint = torch.load(self.model_save_path, map_location=self.device)
        print("Checkpoint loaded!")
        self.print_summary(checkpoint)
        self.prepare_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Only load optimizer state for training, not for prediction
        if not self.pred:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded!")

        print("Model loaded!")

        if not self.best_model_path == None:
            self.exp_dir = Path(self.best_model_path).parent
            print("Set exp dir to: ", self.exp_dir)
        

    def load_model_old(self):
        """load (old) model from model_save_path;
        "old" model was saved using torch.save(ENTIRE MODEL, path);
        "new" model was saved using torch.save(MODEL_DICT, path) which requires new load function
        """
        print("Loading model from: ")
        print(self.model_save_path)
        if not self.best_model_path == None:
            self.model = torch.load(self.best_model_path, map_location=torch.device(self.device))
            self.exp_dir = Path(self.best_model_path).parent
            print("Set exp dir to: ", self.exp_dir)
        else:
            self.model = torch.load(self.model_save_path)
        print("Model loaded!")

    def set_pdf_path_pred(self):
        """set path to pdf file for saving predictions and results
        """
        pdf_path = os.path.join(self.exp_dir, "predictions.pdf")

        count = 1
        while os.path.exists(pdf_path):
            pdf_path = os.path.join(self.exp_dir, "predictions_{0}.pdf".format(count))
            count += 1
        self.pdf = PdfPages(pdf_path)

    def test_model(self, valid_dataset=None):
        """predict images from valid_dataset and save results to pdf

        Args:
            valid_dataset (Dataset, optional): image dataset used for testing. Defaults to None.
        """
        if valid_dataset == None:
            valid_dataset = Dataset(
                self.x_test, 
                self.y_test,
                class_values=self.class_values,
                preprocessing=get_preprocessing(self.preprocessing_fn),
            )

        self.set_pdf_path_pred()
        self.load_model()

        for n in range(len(valid_dataset)):
            # n = np.random.choice(len(valid_dataset))
            # image_vis = valid_dataset[n][0].astype('uint8')
            image, gt_mask = valid_dataset[n]

            img_name = Path(self.x_test[n]).name
            img = cv2.imread(self.x_test[n])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
            with torch.no_grad():
                pr_mask = torch.softmax(self.model(x_tensor), dim=1)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
            self.save_images_to_pdf(
                title=img_name,
                image=img, 
                ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )
        self.pdf.close()


    def get_class_entropies(self, img_):
        """calculate class entropies for one image

        Args:
            img_ (str or numpy array): should be either image path or image array

        Returns:
            class_entropies (numpy array): class entropies for image prediction
        """

        if type(img_) == str:
            img = cv2.imread(img_)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = self.preprocessing(image=img)['image']

        elif type(img_) == np.ndarray:
            img = img_
        else:
            print("Error: Wrong input type!")
            return None

        softmax_out = self.get_model_output(img)
        softmax_out = (softmax_out.squeeze().cpu().numpy())
        class_entropies = -np.sum(softmax_out * np.log2(softmax_out + 1e-10), axis=0)

        return class_entropies
    
    def save_uncertainty_images(self, epoch):
        """generate entropy images for all images in test set and save them as png

        Args:
            epoch (int): epoch number
        """
        
        print("Saving entropy images of test set...")
        img_list = self.x_test
        for img_ in img_list:

            entropy_img_name = img_.split("/")[-1].replace(".JPG", "_entropy_epoch_{0}.png".format(epoch))
            entropy_img_path = os.path.join(self.exp_dir, entropy_img_name)
            print(entropy_img_path)

            class_entropies = self.get_class_entropies(img_)

            plt.figure()
            plt.imsave(entropy_img_path, class_entropies)
            plt.close()

    def calculate_uncertainty_scores(self, new_image_num=100, keep=30, split=0.6):
        """ generate entropy scores for images and return image list sorted by entropy

        Args:
            epoch (int, optional): current epoch. Defaults to 0.
            test_set (bool, optional): save images from test set (if true) or return sorted list (if false). Defaults to False.

        Returns:
            _type_: None if test_set is True, sorted list of image paths if test_set is False
        """

        print("Shuffling pool images...")
        np.random.shuffle(self.x_pool)
        print("Pool length: ", len(self.x_pool))
        print("Getting new training samples from pool...")
        if len(self.x_pool) < new_image_num:
            new_image_num = len(self.x_pool)
        
        try:
            new_training_samples = [self.x_pool.pop(0) for i in range(new_image_num)]
        except IndexError:
            print("Pool is empty!")
            return None


        if len(new_training_samples) == 0:
            print("Pool is empty!")
            return None
        print("Pool length: ", len(self.x_pool))
        entropy_scores = np.zeros(len(new_training_samples))

        print("Calculacting uncertaianty scores...")
        for i, img_ in enumerate(new_training_samples):
            class_entropies = self.get_class_entropies(img_)
            total_entropy = np.sum(class_entropies)
            entropy_scores[i] = total_entropy

        print("Sorting images by entropy...")
        sorted_score_indices = np.argsort(entropy_scores)
        sorted_train_imgs = [new_training_samples[i] for i in sorted_score_indices]
        sorted_scores = [entropy_scores[i] for i in sorted_score_indices]

        # print("Uncertainty scores: ")
        # print("-----------------")
        # [print("Image: {0} | Entropy: {1}".format(sorted_train_imgs[i], sorted_scores[i])) for i in range(len(sorted_train_imgs))]

        keep_train_imgs = sorted_train_imgs[:keep]
        self.x_pool += sorted_train_imgs[keep:]
        print("Pool length after update: ", len(self.x_pool))

        return keep_train_imgs


    def get_model_output(self, img):
        x_tensor = torch.from_numpy(img).to(self.device).unsqueeze(0)
        self.model.eval()
        # model returns logits (activation=None); apply softmax here so
        # downstream code (.round(), entropy) sees probabilities.
        with torch.no_grad():
            pr_mask = torch.softmax(self.model(x_tensor), dim=1)

        return pr_mask

    def calculate_prediction(self, img, return_entropy=False):
        """calculate prediction for one image crop

        Args:
            img (np array): image read by opencv (color channels: BGR)

        Returns:
            np array: mask array with class values, color coded with class colors
        """


        pr_mask = self.get_model_output(img)
        pr_mask_e = (pr_mask.squeeze().cpu().numpy())
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        class_entropies = -np.sum(pr_mask_e * np.log2(pr_mask_e + 1e-10), axis=0)
        pr_mask = pr_mask.transpose(1, 2, 0)
        out = np.zeros((pr_mask.shape[0], pr_mask.shape[1]), dtype=np.int64)

        # for v in self.class_values:
            # out[pr_mask[:,:,v] == 1] = v
        
        for channel_idx, class_val in enumerate(self.class_values):
            out[pr_mask[:, :, channel_idx] == 1] = class_val

        if return_entropy:
            return out, class_entropies
        else:
            return out
    
    def save_mask(self, mask, img_name):
        from PIL import Image
        mask_name = "mask.png"

        if img_name.endswith(".JPG"):
            mask_name = img_name.replace(".JPG", "_mask.png")
        elif img_name.endswith(".jpg"):
            mask_name = img_name.replace(".jpg", "_mask.png")

        save_path = os.path.join(self.test_dir, mask_name)
        im = Image.fromarray(mask)
        im.save(save_path)
        
    def save_entropy_image(self, entropies, img_name):

        if img_name.endswith(".JPG"):
            img_name = img_name.replace(".JPG", "_entropy.png")
        elif img_name.endswith(".jpg"):
            img_name = img_name.replace(".jpg", "_entropy.png")

        plt.figure()
        plt.imsave(os.path.join(self.test_dir, img_name), entropies)
        plt.close()

    def predict_whole_image(self, img, debug=False, get_entropies=False):
        """predict whole image by dividing it into 1024x1024 crops and calculate prediction for each crop
            if debug is True, return empty mask for testing

        Args:
            img (np.ndarray or PIL): _description_

        Returns:
            np.ndarray: mask results for whole image (categorical values)
        """
        
        print("predicting...")
        if not type(img) == np.ndarray:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # size = 1024
        original_img = img
        focus_size = 256
        outer_gap = (self.size - focus_size)//2
        shift = focus_size

        print("original img shape: ", original_img.shape)
        width = int(np.ceil(img.shape[1]/self.size) * self.size)
        height = int(np.ceil(img.shape[0]/self.size) * self.size)
        padded_img = np.zeros((height, width, 3), dtype=np.uint8)
        padded_height = height + outer_gap + self.size - (height+outer_gap)%self.size
        padded_width = width + outer_gap + self.size - (height+outer_gap)%self.size

        padded_img_large = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        padded_img_large[outer_gap:img.shape[0]+outer_gap, outer_gap:img.shape[1]+outer_gap, :] = img

        print("HEIGHT / WIDTH: ", height, width)
        print("SIZE PADDED LARGE: ", padded_img_large.shape)

        # width_l = int(np.ceil(padded_img_large.shape[1]/shift) * shift)
        # height_l = int(np.ceil(padded_img_large.shape[0]/shift) * shift)
        # print("WIDTH: ", width_l)
        # print("HEIGHT: ", height_l)

        whole_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
        width_steps = np.arange(int(padded_width/shift))
        height_steps = np.arange(int(padded_height/shift))

        print("WIDTH STEPS: ", width_steps)
        print("HEIGHT STEPS: ", height_steps)

        if debug:
            return whole_mask

        if get_entropies:
            whole_entropies = np.zeros((padded_height, padded_width), dtype=np.float32)

        for w in tqdm(width_steps):
            for h in height_steps:
                h1 = int(h*shift)
                h2 = int(h1 + self.size)
                w1 = int(w*shift)
                w2 = int(w1 + self.size)

                sub_img = padded_img_large[h1:h2, w1:w2, :]
                sub_img = self.preprocessing(image=sub_img)['image']
                # print("Coords: ", h1, h2, w1, w2)
                # print("IMAGE: ", sub_img.shape)

                if get_entropies:
                    color_coded_mask, entropies = self.calculate_prediction(sub_img, return_entropy=True)
                    entropies = entropies[outer_gap:self.size-outer_gap, outer_gap:self.size-outer_gap]
                    print("Entropies:", entropies.shape)
                    whole_entropies[h1+outer_gap:h2-outer_gap, w1+outer_gap:w2-outer_gap] = entropies
                else:
                    color_coded_mask = self.calculate_prediction(sub_img)
                    print("Unique for subImg: ", np.unique(color_coded_mask))
                
                color_coded_mask = color_coded_mask[outer_gap:focus_size + outer_gap, outer_gap:focus_size + outer_gap]
                whole_mask[h1+outer_gap:h2-outer_gap, w1+outer_gap:w2-outer_gap] = color_coded_mask

                # plt.figure()
                # plt.imshow(img)
                # plt.show()
        whole_mask = whole_mask[outer_gap:original_img.shape[0]+outer_gap, outer_gap:original_img.shape[1]+outer_gap]

        if get_entropies:
            whole_entropies = whole_entropies[outer_gap:original_img.shape[0]+outer_gap, outer_gap:original_img.shape[1]+outer_gap]
            return whole_mask, whole_entropies
        else:
            return whole_mask
    
    def predict(self, load_model=True, test_dir=None, model_path=None, save_entropies=False):
        """ calculate predictions for all images in test folder and save it to pdf
        """

        # self.set_pdf_path_pred()

        self.load_model()

        # self.test_dir = Path("C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\test")
        img_paths = [os.path.join(self.test_dir, x) for x in os.listdir(self.test_dir) if x.lower().endswith(".jpg")]
        number_imgs = len(img_paths)
        # img_paths = random.sample(img_paths, 35)
        for n in range(len(img_paths)):
            img_name = Path(img_paths[n]).name
            print("{0} / {1} | {2}".format(n, number_imgs, img_paths[n]))

            try:
                img = cv2.imread(img_paths[n])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if save_entropies:
                    whole_mask, entropies = self.predict_whole_image(img, get_entropies=True)
                    self.save_entropy_image(entropies, img_name)
                else:
                    whole_mask = self.predict_whole_image(img)
                self.save_mask(whole_mask, img_name)
            except Exception as e:
                print("Error: ", e)
                print("FILE: ", img_paths[n])
                print("continue...")
                continue

def check_gpu_availability():
    if torch.cuda.is_available():
        try:
            # Force actual initialization of the assigned device
            torch.cuda.set_device(0)  # device 0 within CUDA_VISIBLE_DEVICES, i.e. GPU 3
            _ = torch.zeros(1).cuda()
            print(f"GPU successfully initialized: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"FATAL: GPU init failed despite cuda.is_available() — {e}")
            print("Assigned GPU problems?. Exiting to avoid silent CPU run.")
            sys.exit(1)
    else:
        print("FATAL: No CUDA available.")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train or test segmentation model')
    parser.add_argument('--mode', default='train', type=str, help='mode: "train" for training model \n "test" for testing with ground truth and save results to pdf file \n "predict" for predicting whole images and save prediction mask', required=False)
    parser.add_argument('--encoders', action="append", type=str, help='list of encoders to train/test/predict', required=False)
    parser.add_argument('--config', default='server_local_debug.yml', type=str, help='config file for training/testing/predicting', required=True)
    parser.add_argument('--cross_val_run', default=0, type=int, help='cross validation run index (0...n)', required=False)
    parser.add_argument('--epochs', default=600, type=int, help='number of training epochs', required=False)
    parser.add_argument('--memory_optimized', action='store_true', help='enable memory optimized training with gradient checkpointing and 8-bit optimizer', required=False)
    parser.add_argument('--accumulation_steps', default=4, type=int, help='number of accumulation steps for gradient accumulation (effective batch = batch_size * accumulation_steps)', required=False)
    parser.add_argument('--data_leakage', action='store_true', help='allow data leakage in cross validation (subimages from one original image might be placed in training and validation set)', required=False)
    parser.add_argument('--batch_size', default=2, type=int, help='batch size for training', required=False)
    parser.add_argument('--early_stopping_patience', default=0, type=int, help='stop training if val IoU does not improve for this many epochs; 0 disables early stopping', required=False)
    parser.add_argument('--num_workers', default=16, type=int, help='number of DataLoader worker processes for data loading; increase to reduce GPU idle time', required=False)
    parser.add_argument('--ignore_background', default=False, action=argparse.BooleanOptionalAction, help='exclude background class from training and mask it out of the loss/metric (default: False; the default path trains background as an explicit class and relies on focal loss to down-weight it)')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='use weighted sampler to address class imbalance in training data', required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # config_file = 'server_config_graz.yml'

    # encoder_list = ['mit_b0', 'efficientnet-b3', 'efficientnet-b7', 'vgg16', 'resnet50']
    # encoder_list = ['mit_b1', 'mit_b3', 'mit_b5']

    check_gpu_availability()

    seed = 10
    n_folds = 5
    train_split = (n_folds - 1)/n_folds
    
    args = parse_args()


    print(args.encoders)
    encoder_list = args.encoders
    default_encoder = 'mit_b5'
    config_file = args.config
    cross_val_run = args.cross_val_run
    cross_val_exp_series = str(uuid.uuid4())[:6]
    memory_optimized = args.memory_optimized
    epochs = args.epochs
    accumulation_steps = args.accumulation_steps
    data_leakage = args.data_leakage
    batch_size = args.batch_size
    ignore_background = args.ignore_background
    use_weighted_sampler = args.use_weighted_sampler
    early_stopping_patience = args.early_stopping_patience
    num_workers = args.num_workers

    if data_leakage:
        print(" CAUTION: \n Data leakage is allowed in cross validation! \n Subimages from one original image might be placed in training and validation set.")
        print("Overfitting might occur and results might be overestimated! \n CAUTION! \n")

    print(encoder_list)

    if args.mode == 'train':
        print("Training mode...")
        print(f"Memory optimized: {memory_optimized}")
        print(f"Number of epochs: {epochs}")
        trainer = Trainer(epochs=epochs,
                          save_val_uncertainty=False,
                          train_split=train_split,
                          cross_validation=True,
                          config_file=config_file,
                          n_folds=n_folds,
                          cross_val_exp_series=cross_val_exp_series,
                          memory_optimized=memory_optimized,
                          batch_size=batch_size,
                          accumulation_steps=accumulation_steps,
                          data_leakage=data_leakage,
                          ignore_background=ignore_background,
                          use_weighted_sampler=use_weighted_sampler,
                          num_workers=num_workers
                          )
        for encoder in encoder_list:
            trainer.set_encoder(encoder) # set encoder for model
            # for cross_val_run in range(1, n_folds+1):
            print("Train: ", encoder, "\n")
            print("Cross Val Run: ", cross_val_run, "\n")
            trainer.set_seed(seed) # set seed for reproducibility
            trainer.set_paths(cross_val_run, train=True) # set paths for training
            
            trainer.create_dataloaders(augmentations=True) # create dataloaders from image and mask paths
            trainer.prepare_model() # create Unet object and loss & metric objects and Training/Validation Epoch Runners
            trainer.train_model(early_stopping_patience=early_stopping_patience) # start training routine using Training/Validation Epoch Runners
            # trainer.test_model()
    
    elif args.mode == 'test':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Test: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seed, config_file=config_file, ignore_background=ignore_background, num_workers=num_workers) # create Trainer object and set default values
        trainer.set_paths() # set paths for testing (not recently tested)
        trainer.test_model() # test model

    elif args.mode == 'predict':
        # for encoder in encoder_list:
        if encoder_list == None:
            print("Using default encoder: ", default_encoder)
            encoder = default_encoder
        elif len(encoder_list) == 1:
            print("Using encoder: ", encoder_list[0])
            encoder = encoder_list[0]
        elif len(encoder_list) > 1:
            print("Too many encoders! Exiting...")
            exit()


        print("Predict: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seed, pred=True, config_file=config_file, ignore_background=ignore_background, num_workers=num_workers)
        trainer.set_paths(pred=True)
        trainer.predict(save_entropies=True)

