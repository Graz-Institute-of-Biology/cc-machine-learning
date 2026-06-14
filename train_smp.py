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
import traceback
import shutil
import datetime


def dataloader_worker_init(worker_id):
    """Disable OpenCV's internal threading inside DataLoader workers.

    Each worker otherwise spawns its own OpenCV thread pool (used by warpAffine
    etc. in the augmentations), oversubscribing the cores allocated to the job —
    notably on SLURM, where the cgroup only has the cores requested via
    --cpus-per-task. One single-threaded OpenCV per worker parallelises cleanly
    across workers instead. Does not affect results, only thread scheduling.
    """
    cv2.setNumThreads(0)


class RecordingWeightedSampler(WeightedRandomSampler):
    """WeightedRandomSampler that records the drawn indices for post-epoch analysis."""

    def __iter__(self):
        self.last_indices = list(super().__iter__())
        return iter(self.last_indices)


class ModelEMA:
    """Exponential moving average of model weights.

    Keeps a shadow copy updated after every optimizer step:
        ema_w = decay * ema_w + (1 - decay) * model_w
    Validation/selection and the saved best checkpoint use the EMA weights —
    the averaged model is typically slightly better and much less noisy
    epoch-to-epoch than the raw weights. Non-float buffers (e.g. BatchNorm
    num_batches_tracked) are copied verbatim.
    """

    def __init__(self, model, decay=0.999):
        import copy
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_sd = model.state_dict()
        for key, ema_val in self.module.state_dict().items():
            model_val = model_sd[key].detach()
            if ema_val.dtype.is_floating_point:
                ema_val.mul_(self.decay).add_(model_val, alpha=1.0 - self.decay)
            else:
                ema_val.copy_(model_val)


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
    """Argmax-based IoU that ignores background pixels (where all target channels
    are zero). Used when ignore_background=True (no background channel in the model).

    argmax assigns every pixel to exactly one class, so no pixels are dropped the
    way threshold-0.5 on softmax drops low-confidence (typically rare-class) pixels
    where no channel exceeds 0.5. Background pixels then have predictions zeroed
    out so they cannot become false positives.
    Expects y_pred as logits — argmax is invariant to softmax so it's skipped.
    """
    __name__ = 'iou_score'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        pred_idx = y_pred.argmax(dim=1)
        pred_onehot = torch.nn.functional.one_hot(
            pred_idx, num_classes=y_pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        fg_mask = (y_true.sum(dim=1, keepdim=True) > 0).float()  # (B, 1, H, W)
        return self._iou(pred_onehot * fg_mask, y_true)


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


class DiceCEWeighted(torch.nn.Module):
    """Compound Dice + class-weighted cross-entropy on logits.

    Dice operates at the segment level so rare classes can't be ignored by
    predicting them as background; class-weighted CE adds per-pixel calibration
    with emphasis on rare classes. ce_ignore_index skips a class (e.g. background)
    in CE; dice_classes restricts Dice to a subset of channels so the two
    components stay consistent when background is ignored.
    Expects y_pred as logits; one-hot targets are argmax'd to class indices.
    """
    __name__ = 'DiceCE_loss'

    def __init__(self, class_weights, dice_weight=0.5, ce_weight=0.5,
                 ce_ignore_index=-100, dice_classes=None, label_smoothing=0.0):
        super().__init__()
        self._dice = smp.losses.DiceLoss(
            mode='multiclass', from_logits=True, classes=dice_classes
        )
        # label_smoothing > 0 makes CE tolerant of mislabeled pixels (e.g. the
        # cyanosbark<->cyanosmoss ground-truth ambiguity) by capping the target
        # confidence; 0.0 = standard CE.
        self._ce = torch.nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ce_ignore_index,
            label_smoothing=label_smoothing
        )
        self.dice_w = dice_weight
        self.ce_w = ce_weight

    def forward(self, y_pred, y_true):
        y_true_idx = y_true.argmax(dim=1).long()
        return self.dice_w * self._dice(y_pred, y_true_idx) + self.ce_w * self._ce(y_pred, y_true_idx)


class TverskyCEWeighted(torch.nn.Module):
    """Compound Tversky + class-weighted cross-entropy on logits.

    Tversky generalizes Dice with separate FP/FN weights:
      loss = 1 - TP / (TP + alpha*FP + beta*FN)
      alpha=beta=0.5  → Dice
      beta > alpha    → penalises false negatives harder, raising recall on rare
                        classes (the binding constraint when classes are visually
                        similar, e.g. cyanosbark vs cyanosmoss).
    Combined with class-weighted CE for per-pixel calibration. Background
    handling (ce_ignore_index, tversky_classes) mirrors DiceCEWeighted so the
    two loss components stay consistent.
    Expects y_pred as logits; one-hot targets are argmax'd to class indices.
    """
    __name__ = 'TverskyCE_loss'

    def __init__(self, class_weights, tversky_weight=0.5, ce_weight=0.5,
                 alpha=0.3, beta=0.7,
                 ce_ignore_index=-100, tversky_classes=None, label_smoothing=0.0):
        super().__init__()
        self._tversky = smp.losses.TverskyLoss(
            mode='multiclass', from_logits=True, classes=tversky_classes,
            alpha=alpha, beta=beta,
        )
        self._ce = torch.nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ce_ignore_index,
            label_smoothing=label_smoothing
        )
        self.t_w = tversky_weight
        self.ce_w = ce_weight

    def forward(self, y_pred, y_true):
        y_true_idx = y_true.argmax(dim=1).long()
        return self.t_w * self._tversky(y_pred, y_true_idx) + self.ce_w * self._ce(y_pred, y_true_idx)


class IoUForegroundOnly(torch.nn.Module):
    """Argmax-based IoU over foreground channels only (excludes channel 0 = background).
    Pixels predicted as background simply don't appear in any FG channel — counted
    as FN for the true FG class with no FP, which is the correct behavior.

    Argmax avoids the threshold-0.5 trap on multiclass softmax: with C classes,
    softmax peaks often sit at 0.3-0.5 on ambiguous (typically rare-class) pixels,
    so threshold-0.5 silently drops them from both TP and FP counts.
    Expects y_pred as logits — argmax is invariant to softmax so it's skipped.
    """
    __name__ = 'iou_score'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        pred_idx = y_pred.argmax(dim=1)
        pred_onehot = torch.nn.functional.one_hot(
            pred_idx, num_classes=y_pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        return self._iou(pred_onehot[:, 1:], y_true[:, 1:])


class IoUBackgroundOnly(torch.nn.Module):
    """Argmax-based IoU over the background channel only (channel 0).
    Reported alongside the foreground IoU so the model's background-vs-foreground
    behavior is visible without contaminating the main model-selection metric.
    Expects y_pred as logits — argmax is invariant to softmax so it's skipped.
    """
    __name__ = 'iou_background'

    def __init__(self, threshold=0.5):
        super().__init__()
        self._iou = smp_metrics.IoU(threshold=threshold)

    def forward(self, y_pred, y_true):
        pred_idx = y_pred.argmax(dim=1)
        pred_onehot = torch.nn.functional.one_hot(
            pred_idx, num_classes=y_pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        return self._iou(pred_onehot[:, :1], y_true[:, :1])


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
                 num_workers=4,
                 loss_name='focal',
                 class_weight_power=0.5,
                 bg_weight_multiplier=1.0,
                 save_confusion_each_epoch=False,
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 hard_sample_target=None,
                 hard_sample_strength=2.0,
                 hard_sample_ema=0.7,
                 lr=1e-4,
                 decoder_lr_mult=1.0,
                 lr_schedule='plateau',
                 warmup_epochs=3,
                 use_amp=False,
                 grad_checkpoint=False,
                 label_smoothing=0.0,
                 ema_decay=0.0,
                 aug_photometric=False,
                 aug_scale_limit=0.1,
                 ):

        self.size = size
        self.pred = pred
        self.seed = seed
        self.dataset = None
        self.research_site = None
        self.save_val_uncertainty = save_val_uncertainty
        self.ignore_background = ignore_background
        self.use_weighted_sampler = use_weighted_sampler
        self.loss_name = loss_name
        self.class_weight_power = class_weight_power
        self.bg_weight_multiplier = bg_weight_multiplier
        self.save_confusion_each_epoch = save_confusion_each_epoch
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.hard_sample_target = hard_sample_target
        self.hard_sample_strength = hard_sample_strength
        self.hard_sample_ema = hard_sample_ema
        self.hard_sample_target_idx = None  # resolved in create_dataloaders
        self.use_amp = use_amp
        self.grad_checkpoint = grad_checkpoint
        self.label_smoothing = label_smoothing
        self.ema_decay = ema_decay
        self.ema = None  # ModelEMA instance, created in prepare_model when ema_decay > 0
        self.aug_photometric = aug_photometric
        self.aug_scale_limit = aug_scale_limit
        self.decoder_lr_mult = decoder_lr_mult
        self.warmup_epochs = warmup_epochs
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
        self.lr = lr
        self.uncertainy_runs = 10
        self.weight_decay = 1e-4
        # 'plateau' = legacy ReduceLROnPlateau on val IoU (default, reproduces
        # all past runs). 'cosine' = linear warmup + cosine decay over the fixed
        # epoch budget, the standard recipe for transformer (mit_*) encoders.
        self.lr_scheduler = "ReduceLROnPlateau" if lr_schedule == 'plateau' else "WarmupCosine"

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
                "warmup_epochs": self.warmup_epochs,
                "decoder_lr_mult": self.decoder_lr_mult,
                "accumulation_steps": self.accumulation_steps,
                "amp": self.use_amp,
                "grad_checkpoint": self.grad_checkpoint,
                "label_smoothing": self.label_smoothing,
                "ema_decay": self.ema_decay,
                "aug_photometric": self.aug_photometric,
                "aug_scale_limit": self.aug_scale_limit,
                
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

        # permutation (without replacement) — np.random.choice's default
        # replacement=True duplicated tiles in train AND leaked tiles into both
        # train and val at the same time.
        indices = np.random.permutation(len(self.images_fps))
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


    def compute_per_image_class_presence(self, unique_imgs):
        """Binary class-presence matrix (N_images × N_classes) built from mask tiles.

        An image is marked present for class c if any of its tiles contain ≥1 pixel
        of c. Used by iterative_stratification_split to balance rare classes across
        CV folds before the train/val split is determined.
        """
        print("\nComputing per-image class presence for stratified splitting...")
        img_to_idx = {img: i for i, img in enumerate(unique_imgs)}
        presence = np.zeros((len(unique_imgs), len(self.class_values)), dtype=np.float32)

        for tile_fn in self.ids:
            stem = tile_fn.split("_part")[0]
            if stem not in img_to_idx:
                continue
            mask_fn = tile_fn.replace(".JPG", ".png").replace(".jpg", ".png")
            mask = cv2.imread(str(os.path.join(self.mask_dir, mask_fn)), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            img_idx = img_to_idx[stem]
            for cls_idx, cls_val in enumerate(self.class_values):
                if np.any(mask == cls_val):
                    presence[img_idx, cls_idx] = 1.0

        print(f"{'Class':<20} {'Images with class':>18} {'%':>6}")
        print("-" * 48)
        for cls_idx, label in enumerate(self.labels):
            n = int(presence[:, cls_idx].sum())
            print(f"{label:<20} {n:>18} {100 * n / len(unique_imgs):>5.1f}%")
        return presence

    def iterative_stratification_split(self, imgs, presence):
        """Assign images to k folds using iterative stratification (Sechidis et al. 2011).

        Returns (train_imgs, val_imgs) for self.cross_val_run. Each fold receives a
        proportional share of every class's image-level prevalence, including the
        rarest ones. This prevents the skewed ratios (e.g. barkdominated T/V = 0.33×)
        that occur with random shuffling when class coverage is sparse.
        """
        N, C = presence.shape
        fold_assignments = np.full(N, -1, dtype=int)
        # Desired number of class-c images per fold, initialised from global counts.
        desired = np.tile(presence.sum(axis=0) / self.n_folds, (self.n_folds, 1))  # (k, C)
        remaining = list(range(N))

        while remaining:
            counts = presence[remaining].sum(axis=0)
            active = np.where(counts > 0)[0]

            if len(active) == 0:
                # Only unlabelled (all-zero) images left — balance by fold size.
                for i in remaining:
                    sizes = np.bincount(fold_assignments[fold_assignments >= 0],
                                        minlength=self.n_folds)
                    fold_assignments[i] = int(np.argmin(sizes))
                break

            # Process the rarest active class first (minimises label imbalance).
            rarest = int(active[np.argmin(counts[active])])
            class_imgs = [i for i in remaining if presence[i, rarest]]

            for i in class_imgs:
                target = int(np.argmax(desired[:, rarest]))
                fold_assignments[i] = target
                remaining.remove(i)
                for c in range(C):
                    if presence[i, c]:
                        desired[target, c] = max(0.0, desired[target, c] - 1.0)

        val_imgs   = [imgs[i] for i in range(N) if fold_assignments[i] == self.cross_val_run]
        train_imgs = [imgs[i] for i in range(N) if fold_assignments[i] != self.cross_val_run]

        # Report per-class image-level balance so fold quality is visible in the log.
        print(f"\nStratified split — fold {self.cross_val_run}/{self.n_folds}: "
              f"{len(train_imgs)} train / {len(val_imgs)} val images")
        print(f"{'Class':<20} {'Train imgs':>10} {'Val imgs':>9} {'T/V ratio':>10}")
        print("-" * 52)
        train_set, val_set = set(train_imgs), set(val_imgs)
        for cls_idx, label in enumerate(self.labels):
            t = sum(1 for i, img in enumerate(imgs) if img in train_set and presence[i, cls_idx])
            v = sum(1 for i, img in enumerate(imgs) if img in val_set   and presence[i, cls_idx])
            ratio = t / v if v > 0 else float('inf')
            print(f"{label:<20} {t:>10} {v:>9} {ratio:>10.2f}x")
        print()
        return train_imgs, val_imgs

    def split_train_val(self, imgs):
        """Split imgs into (train_imgs, val_imgs) for self.cross_val_run.

        Uses iterative stratification when per-image class presence has been
        computed (cross-validation mode). Falls back to the original contiguous-
        slice random split for non-CV or legacy paths.
        """
        if hasattr(self, '_image_presence'):
            presence = np.array([
                self._image_presence.get(img, np.zeros(len(self.class_values)))
                for img in imgs
            ])
            return self.iterative_stratification_split(imgs, presence)

        # Original random-slice fallback.
        val_fold_size = len(imgs) // self.n_folds
        val_start = self.cross_val_run * val_fold_size
        val_end   = val_start + val_fold_size if self.cross_val_run < self.n_folds - 1 else len(imgs)
        unique_val_imgs   = imgs[val_start:val_end]
        unique_train_imgs = imgs[:val_start] + imgs[val_end:]
        print(f"Val start: {val_start}  Val end: {val_end}  "
              f"Train: {len(unique_train_imgs)}  Val: {len(unique_val_imgs)}")
        return unique_train_imgs, unique_val_imgs


    def get_mixed_train_val_imgs(self, sorted_unique_imgs):
        # Split per research site so each site's class balance is preserved
        # independently. Shuffling is omitted — iterative_stratification_split
        # assigns by label, not position, so pre-shuffling has no effect.
        campina_imgs    = [x for x in sorted_unique_imgs if x.split("_")[1] == "C"]
        terra_firme_imgs = [x for x in sorted_unique_imgs if x.split("_")[1] == "TF"]

        tf_train, tf_val = self.split_train_val(terra_firme_imgs)
        c_train,  c_val  = self.split_train_val(campina_imgs)

        return tf_train + c_train, tf_val + c_val

    def get_mixed_snow_train_val_imgs(self, sorted_unique_imgs):
        snow_imgs     = [x for x in sorted_unique_imgs if "september" in x]
        non_snow_imgs = [x for x in sorted_unique_imgs if "september" not in x]

        ns_train, ns_val = self.split_train_val(non_snow_imgs)
        s_train,  s_val  = self.split_train_val(snow_imgs)

        return ns_train + s_train, ns_val + s_val

    def get_train_val_imgs(self, sorted_unique_imgs, train_split):
        if self.cross_validation:
            # Stratified split: ordering is irrelevant, stratification assigns by label.
            return self.split_train_val(sorted_unique_imgs)
        else:
            # Non-CV holdout: random shuffle then slice.
            shuffled = sorted_unique_imgs.copy()
            np.random.shuffle(shuffled)
            train_len = int(len(shuffled) * train_split)
            return shuffled[:train_len], shuffled[train_len:]

    def set_unique_paths(self, train_split, sorted_unique_imgs):

        if self.research_site == "mixed":
            unique_train_imgs, unique_val_imgs = self.get_mixed_train_val_imgs(sorted_unique_imgs)
        
        
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



    def snapshot_run_artifacts(self):
        """Copy the code + config that produced this run into exp_dir/code_snapshot/.

        Makes each experiment self-contained: the training script, the resolved
        config YAML, and the ontology JSON are archived alongside the model and
        logs, plus a run_config.json capturing the resolved hyperparameters,
        full command line, and git commit. This replaces the old practice of
        freezing a whole separate copy of the script per project.

        Best-effort: any individual copy failure is logged, not raised, so a
        snapshot problem never aborts a training run.
        """
        snap_dir = os.path.join(self.exp_dir, "code_snapshot")
        try:
            os.makedirs(snap_dir, exist_ok=True)

            # 1. The training script and its local helper modules. The helpers
            # define the augmentation pipeline and dataset loading, so they are
            # part of what makes a run reproducible.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for src in [os.path.abspath(__file__),
                        os.path.join(script_dir, "training_helper.py"),
                        os.path.join(script_dir, "smp_dataset.py")]:
                try:
                    shutil.copy2(src, snap_dir)
                except Exception as e:
                    print(f"[snapshot] could not copy {src}: {e}")

            # 2. Config YAML and ontology JSON (relative to cwd or absolute).
            for src in [getattr(self, "config_file", None),
                        getattr(self, "ontology_file_name", None)]:
                if src and os.path.isfile(src):
                    try:
                        shutil.copy2(src, snap_dir)
                    except Exception as e:
                        print(f"[snapshot] could not copy {src}: {e}")

            # 3. Resolved run config — the values that actually drove the run.
            run_config = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "git_commit": self.get_git_commit_hash(),
                "command_line": " ".join(sys.argv),
                "exp_dir": str(self.exp_dir),
                "encoder": self.encoder,
                "encoder_weights": self.encoder_weights,
                "seed": self.seed,
                "size": self.size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "accumulation_steps": self.accumulation_steps,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "optimizer": self.optimizer_name,
                "lr_scheduler": self.lr_scheduler,
                "loss_name": self.loss_name,
                "class_weight_power": self.class_weight_power,
                "bg_weight_multiplier": self.bg_weight_multiplier,
                "tversky_alpha": self.tversky_alpha,
                "tversky_beta": self.tversky_beta,
                "hard_sample_target": self.hard_sample_target,
                "hard_sample_strength": self.hard_sample_strength,
                "hard_sample_ema": self.hard_sample_ema,
                "decoder_lr_mult": self.decoder_lr_mult,
                "warmup_epochs": self.warmup_epochs,
                "use_amp": self.use_amp,
                "grad_checkpoint": self.grad_checkpoint,
                "label_smoothing": self.label_smoothing,
                "ema_decay": self.ema_decay,
                "aug_photometric": self.aug_photometric,
                "aug_scale_limit": self.aug_scale_limit,
                "ignore_background": self.ignore_background,
                "use_weighted_sampler": self.use_weighted_sampler,
                "memory_optimized": self.memory_optimized,
                "cross_validation": self.cross_validation,
                "cross_val_run": self.cross_val_run,
                "n_folds": self.n_folds,
                "num_workers": self.num_workers,
                "config_file": getattr(self, "config_file", None),
                "ontology_file": getattr(self, "ontology_file_name", None),
                "labels": getattr(self, "labels", None),
                "class_values": getattr(self, "class_values", None),
            }
            with open(os.path.join(snap_dir, "run_config.json"), "w") as f:
                json.dump(run_config, f, indent=2)

            print(f"[snapshot] code + config archived to {snap_dir}")
        except Exception as e:
            print(f"[snapshot] failed (training continues): {e}")


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

            # Snapshot the exact code + config that produced this run into the
            # experiment folder, so a run is self-contained and reproducible
            # without relying on a frozen copy of the script elsewhere.
            self.snapshot_run_artifacts()

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
            if self.cross_validation:
                presence = self.compute_per_image_class_presence(sorted_unique_imgs)
                self._image_presence = {
                    img: presence[i] for i, img in enumerate(sorted_unique_imgs)
                }
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
        self.train_class_counts = train_class_counts  # used for loss class weights

        # Per-tile sampling weight = expected inverse-sqrt-frequency over the
        # tile's pixels. sqrt softens the ratio between rare and common classes
        # (vs. raw 1/freq where cyanosbark@0.27% was ~370x background, now ~15x),
        # reducing the risk of memorizing the few tiles containing the rarest
        # classes while still oversampling them.
        global_inv_freq = 1.0 / np.sqrt(np.maximum(train_class_counts, 1.0))
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
            training_augmentation = get_training_augmentation(
                min_height=1024, min_width=1024,
                photometric=self.aug_photometric,
                scale_limit=self.aug_scale_limit,
            )
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

        loader_kwargs = dict(pin_memory=True, prefetch_factor=4, persistent_workers=True,
                             worker_init_fn=dataloader_worker_init) if self.num_workers > 0 else {}

        if self.use_weighted_sampler:
            sampler = RecordingWeightedSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )
        else:
            sampler = None

        self.sampler = sampler
        
        # Hard-sampling state: base weights stay immutable so EMA-tracked
        # per-tile hardness can multiply against them each epoch without
        # compounding. Hardness starts at zero (= baseline weighting).
        self.base_sample_weights = np.array(self.sample_weights, dtype=np.float64)
        self.tile_hardness = np.zeros(len(self.sample_weights), dtype=np.float64)
        if self.hard_sample_target is not None:
            if sampler is None:
                raise ValueError(
                    "--hard_sample_target requires --use_weighted_sampler (a "
                    "WeightedRandomSampler is needed to bias tile draws)."
                )
            if self.hard_sample_target not in self.labels:
                raise ValueError(
                    f"hard_sample_target='{self.hard_sample_target}' not in labels {self.labels}"
                )
            self.hard_sample_target_idx = self.labels.index(self.hard_sample_target)
            print(f"Hard sampling ON: target='{self.hard_sample_target}' "
                  f"(channel {self.hard_sample_target_idx}), "
                  f"strength={self.hard_sample_strength}, ema={self.hard_sample_ema}")

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



    def enable_mit_gradient_checkpointing(self):
        """Checkpoint every transformer block in the MiT encoder.

        The previous implementation patched encoder.block1..4.forward, but
        smp's MixVisionTransformer.forward_features iterates those ModuleLists
        directly (`for blk in self.block1: x = blk(x, H, W)`) and never calls
        ModuleList.forward — so it silently did nothing, which is why
        --memory_optimized never allowed a larger batch (slurm-5191111.out:
        identical 38.7GB OOM with and without it). Patching each block's own
        forward actually intercepts the calls.

        Memory effect: only block-boundary activations are kept; the attention
        matrices (the dominant tensors at 1024², kept fp32 by autocast) are
        recomputed during backward. With use_reentrant=False the recompute
        replays under the ambient autocast state, so it composes with --amp,
        and non-tensor args (H, W) are supported. The torch.is_grad_enabled()
        guard skips checkpointing overhead during no_grad validation/inference.
        """
        from torch.utils.checkpoint import checkpoint

        n_wrapped = 0
        for block_name in ['block1', 'block2', 'block3', 'block4']:
            block_list = getattr(self.model.encoder, block_name, None)
            if block_list is None:
                continue
            for blk in block_list:
                orig_fwd = blk.forward

                def make_ckpt_forward(fwd):
                    def ckpt_forward(x, H, W):
                        if torch.is_grad_enabled():
                            return checkpoint(fwd, x, H, W, use_reentrant=False)
                        return fwd(x, H, W)
                    return ckpt_forward

                blk.forward = make_ckpt_forward(orig_fwd)
                n_wrapped += 1

        print(f"Gradient checkpointing enabled for MiT encoder "
              f"({n_wrapped} transformer blocks wrapped)")

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

        if self.pred:
            # Inference path: no loss/optimizer/scheduler/epoch runners needed.
            # Skips the dice_ce branch's dependency on self.train_class_counts,
            # which is only populated by create_dataloaders during training.
            self.setup_pred_logging()
            return

        if self.ignore_background:
            self.metrics = [IoUIgnoreBackground(threshold=0.5)]
        else:
            self.metrics = [IoUForegroundOnly(threshold=0.5), IoUBackgroundOnly(threshold=0.5)]

        if self.loss_name == 'focal':
            self.loss = FocalLossIgnoreBackground() if self.ignore_background else MultiClassFocalLoss()
        elif self.loss_name == 'dice_ce':
            # Per-class weights: 1 / freq^p normalized so mean weight = 1.
            # p=0.5 (sqrt) is mild; p=0.75 pushes rare classes harder; p=1.0 is
            # full inverse-frequency and can be unstable with tiny classes.
            freq = self.train_class_counts / max(self.train_class_counts.sum(), 1.0)
            w = 1.0 / np.power(freq + 1e-8, self.class_weight_power)
            w = w / w.mean()

            # Background handling: multiplier < 1 downweights, 0 hard-ignores.
            # Hard-ignore also drops background from Dice so both loss components
            # agree. Only applies when background (value=0) is actually a channel.
            bg_idx = self.class_values.index(0) if 0 in self.class_values else None
            hard_ignore_bg = bg_idx is not None and self.bg_weight_multiplier == 0.0
            if bg_idx is not None:
                w[bg_idx] *= self.bg_weight_multiplier

            print(f"CE class weights (p={self.class_weight_power}, "
                  f"bg_mult={self.bg_weight_multiplier}): "
                  f"{dict(zip(self.labels, np.round(w, 3)))}")
            class_weights_t = torch.tensor(w, dtype=torch.float32, device=self.device)

            if hard_ignore_bg:
                ce_ignore_index = bg_idx
                dice_classes = [i for i in range(len(self.class_values)) if i != bg_idx]
                print(f"Hard-ignoring background: CE ignore_index={bg_idx}, "
                      f"Dice classes={dice_classes}")
            else:
                ce_ignore_index = -100  # nn.CrossEntropyLoss default (no skipping)
                dice_classes = None

            self.loss = DiceCEWeighted(
                class_weights=class_weights_t,
                ce_ignore_index=ce_ignore_index,
                dice_classes=dice_classes,
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_name == 'tversky_ce':
            # Same class-weight + bg handling as dice_ce; only the segment-level
            # loss term swaps Dice → Tversky to expose the FP/FN trade-off via
            # alpha/beta (beta>alpha pushes recall on rare classes).
            freq = self.train_class_counts / max(self.train_class_counts.sum(), 1.0)
            w = 1.0 / np.power(freq + 1e-8, self.class_weight_power)
            w = w / w.mean()
            bg_idx = self.class_values.index(0) if 0 in self.class_values else None
            hard_ignore_bg = bg_idx is not None and self.bg_weight_multiplier == 0.0
            if bg_idx is not None:
                w[bg_idx] *= self.bg_weight_multiplier

            print(f"Tversky alpha={self.tversky_alpha}, beta={self.tversky_beta}; "
                  f"CE class weights (p={self.class_weight_power}, "
                  f"bg_mult={self.bg_weight_multiplier}): "
                  f"{dict(zip(self.labels, np.round(w, 3)))}")
            class_weights_t = torch.tensor(w, dtype=torch.float32, device=self.device)

            if hard_ignore_bg:
                ce_ignore_index = bg_idx
                tversky_classes = [i for i in range(len(self.class_values)) if i != bg_idx]
                print(f"Hard-ignoring background: CE ignore_index={bg_idx}, "
                      f"Tversky classes={tversky_classes}")
            else:
                ce_ignore_index = -100
                tversky_classes = None

            self.loss = TverskyCEWeighted(
                class_weights=class_weights_t,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
                ce_ignore_index=ce_ignore_index,
                tversky_classes=tversky_classes,
                label_smoothing=self.label_smoothing,
            )
        else:
            raise ValueError(f"Unknown loss_name: {self.loss_name}")


        if self.optimizer_name == "AdamW":
            # Transformer-encoder recipe: decoupled weight decay, no decay on
            # biases/norm params (1-d tensors), and an optionally higher LR for
            # the randomly-initialised decoder vs the pretrained encoder.
            param_groups = []
            for prefix, lr in [("encoder.", self.lr),
                               ("", self.lr * self.decoder_lr_mult)]:
                decay_params, no_decay_params = [], []
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    is_encoder = name.startswith("encoder.")
                    if (prefix == "encoder.") != is_encoder:
                        continue
                    (no_decay_params if p.ndim <= 1 else decay_params).append(p)
                if decay_params:
                    param_groups.append(dict(params=decay_params, lr=lr,
                                             weight_decay=self.weight_decay))
                if no_decay_params:
                    param_groups.append(dict(params=no_decay_params, lr=lr,
                                             weight_decay=0.0))
            self.optimizer = torch.optim.AdamW(param_groups)
            print(f"AdamW: encoder lr={self.lr}, decoder lr={self.lr * self.decoder_lr_mult}, "
                  f"wd={self.weight_decay} (none on biases/norms)")
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
                patience=10,          # Wait 10 epochs before reducing
                verbose=True
            )
        elif self.lr_scheduler == "WarmupCosine":
            # Linear warmup then cosine decay over the fixed epoch budget;
            # stepped once per epoch. The multiplier applies per param group,
            # so encoder/decoder differential LRs are preserved.
            warmup = max(0, self.warmup_epochs)

            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, self.epochs - warmup)
                return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            print(f"WarmupCosine schedule: {warmup} warmup epochs, "
                  f"cosine decay over {self.epochs} epochs")

        # EMA shadow model: validation, model selection and the saved best
        # checkpoint all use the averaged weights when enabled.
        if self.ema_decay > 0:
            if self.accumulation_steps == 0:
                raise ValueError("--ema_decay requires the gradient-accumulation "
                                 "training loop (accumulation_steps > 0).")
            self.ema = ModelEMA(self.model, decay=self.ema_decay)
            print(f"Model EMA enabled: decay={self.ema_decay}")

        # Gradient checkpointing — must come AFTER the EMA deepcopy: the patch
        # installs per-block forward closures bound to this model's blocks, and
        # deepcopy copies function attributes by reference, so a copy made
        # after patching would silently run the original model's weights.
        if (self.memory_optimized or self.grad_checkpoint) and 'mit_' in self.encoder:
            self.enable_mit_gradient_checkpointing()

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

        self.setup_logging()


    def save_model(self, epoch, update_latest=False):
        """save model to model_save_path (experiment folder + best_model.pth)
        train/valid dataloaders seem to need a lot of memory and are not saved in the model
        """
        old_ontology_dict = {class_name: data["color"] for class_name, data in self.ontology["ontology"].items()}
        # With EMA, the averaged weights are what was validated/selected, so
        # they are what gets checkpointed.
        model_for_ckpt = self.ema.module if self.ema is not None else self.model
        ckpt_dict = {
            'model_state_dict': model_for_ckpt.state_dict(),
            'ema': self.ema is not None,
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
            is_best = self.valid_logs['iou_score'] > max_score
            if is_best:
                print('New top score: {0} > {1} '.format(self.valid_logs['iou_score'], max_score))
                max_score = self.valid_logs['iou_score']

                self.save_model(epoch=epoch)
                self.save_loss_plots()

                if self.save_val_uncertainty:
                    self.save_uncertainty_images(epoch)

            if self.save_confusion_each_epoch or is_best:
                self.save_confusion_matrix(epoch=epoch)

            self.train_log_df.loc[epoch] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            # valid_log_df has an iou_corpus_fg column (filled by the
            # accumulation loop); this loop doesn't compute it, so pad with NaN
            # to keep the row width consistent.
            self.valid_log_df.loc[epoch] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score'], np.nan]

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

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.valid_logs['iou_score'])
            else:
                self.scheduler.step()



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

            # Hard-sampling per-epoch accumulators (target-class FN/pixel sums
            # indexed by *dataset tile index*; sampler may draw a tile multiple
            # times so we aggregate and divide at end of epoch).
            hs_active = self.hard_sample_target_idx is not None
            if hs_active:
                epoch_tile_fn = np.zeros(len(self.x_train), dtype=np.float64)
                epoch_tile_target_px = np.zeros(len(self.x_train), dtype=np.float64)
                sampler_indices = None  # captured on first batch (after DataLoader iter)

            pbar = tqdm(self.train_loader, desc='train')

            print("cuda available: ", torch.cuda.is_available())
            print(f"[GPU] Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB | "
            f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)

                # bf16 autocast: ~halves activation memory and speeds up matmuls
                # on Ampere+ GPUs with no loss-scaling needed (bf16 has fp32's
                # exponent range). Params/grads stay fp32. No-op when disabled.
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=self.use_amp):
                    pred = self.model(x)
                    loss = self.loss(pred, y)
                (loss / self.accumulation_steps).backward()

                train_loss_sum += loss.item()
                with torch.no_grad():
                    for m in self.metrics:
                        train_metric_sums[m.__name__] += m(pred.detach(), y).item()

                    if hs_active:
                        # Sampler's last_indices is populated when DataLoader
                        # starts iterating; safe to read once we're inside the
                        # loop. Snapshot it on the first batch.
                        if sampler_indices is None:
                            sampler_indices = list(self.sampler.last_indices)
                        t = self.hard_sample_target_idx
                        pred_idx = pred.argmax(dim=1)             # (B, H, W)
                        target_truth = y[:, t]                    # (B, H, W) one-hot for target
                        wrong = (pred_idx != t).float()
                        per_fn = (target_truth * wrong).sum(dim=(1, 2)).cpu().numpy()
                        per_px = target_truth.sum(dim=(1, 2)).cpu().numpy()
                        bs = x.shape[0]
                        start = batch_idx * self.batch_size
                        for bi in range(bs):
                            tile_idx = sampler_indices[start + bi]
                            epoch_tile_fn[tile_idx] += per_fn[bi]
                            epoch_tile_target_px[tile_idx] += per_px[bi]

                postfix = {'loss': train_loss_sum/(batch_idx+1)}
                for name, total in train_metric_sums.items():
                    postfix[name] = total / (batch_idx + 1)
                postfix['gpu_mem'] = f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB"
                pbar.set_postfix(**postfix)

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(self.model)

            # === HARD SAMPLING UPDATE ===
            # End-of-epoch: blend this epoch's per-tile FN rate into the EMA
            # hardness scores, then rebuild sampler weights for the next epoch.
            # Tiles with no target-class pixels this epoch contribute 0 (their
            # previous EMA hardness decays toward 0).
            hs_stats = None
            if hs_active:
                seen = epoch_tile_target_px > 0
                current_hardness = np.zeros_like(epoch_tile_target_px)
                current_hardness[seen] = epoch_tile_fn[seen] / epoch_tile_target_px[seen]
                self.tile_hardness = (
                    self.hard_sample_ema * self.tile_hardness
                    + (1.0 - self.hard_sample_ema) * current_hardness
                )
                new_weights = self.base_sample_weights * (
                    1.0 + self.hard_sample_strength * self.tile_hardness
                )
                self.sampler.weights = torch.as_tensor(new_weights, dtype=torch.double)
                hs_stats = {
                    "hardsample/tiles_with_target_seen": int(seen.sum()),
                    "hardsample/mean_target_fn_rate": float(current_hardness[seen].mean()) if seen.any() else 0.0,
                    "hardsample/max_hardness_ema": float(self.tile_hardness.max()),
                    "hardsample/max_weight_multiplier": float(
                        (1.0 + self.hard_sample_strength * self.tile_hardness).max()
                    ),
                }
                print(f"Hard sampling ep{epoch}: "
                      f"tiles_seen={hs_stats['hardsample/tiles_with_target_seen']}, "
                      f"mean_FN={hs_stats['hardsample/mean_target_fn_rate']:.3f}, "
                      f"max_mult={hs_stats['hardsample/max_weight_multiplier']:.2f}")

            # === VALIDATION ===
            # With EMA enabled, validation/selection runs on the averaged
            # weights — the same weights that get saved as best_model.pth.
            eval_model = self.ema.module if self.ema is not None else self.model
            self.model.eval()
            eval_model.eval()
            val_loss_sum = 0
            val_metric_sums = {m.__name__: 0.0 for m in self.metrics}
            # Corpus-wide per-class IoU: accumulate TP/FP/FN across the full
            # val set and compute IoU once at the end (this is NOT the same as
            # averaging per-batch per-class IoUs, which biases toward classes
            # that happen to land in batches with lots of pixels).
            C = len(self.class_values)
            tp_sum = torch.zeros(C, device=self.device)
            fp_sum = torch.zeros(C, device=self.device)
            fn_sum = torch.zeros(C, device=self.device)

            with torch.no_grad():
                for x, y in tqdm(self.valid_loader, desc='valid'):
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                        enabled=self.use_amp):
                        pred = eval_model(x)
                        loss = self.loss(pred, y)
                    val_loss_sum += loss.item()
                    for m in self.metrics:
                        val_metric_sums[m.__name__] += m(pred, y).item()
                    # Argmax-based one-hot so each pixel contributes to exactly
                    # one class; consistent with the IoU metric wrappers above
                    # and avoids the threshold-0.5 dropout on low-peak softmax.
                    pred_idx = pred.argmax(dim=1)
                    binary = torch.nn.functional.one_hot(
                        pred_idx, num_classes=C
                    ).permute(0, 3, 1, 2).float()
                    tp_sum += (binary * y).sum(dim=(0, 2, 3))
                    fp_sum += (binary * (1 - y)).sum(dim=(0, 2, 3))
                    fn_sum += ((1 - binary) * y).sum(dim=(0, 2, 3))

            per_class_iou = (tp_sum / (tp_sum + fp_sum + fn_sum + 1e-7)).cpu().numpy()

            # Corpus-wide foreground mean IoU. This is the model-selection signal:
            # per-batch averaged IoU (val_metric_sums / len) is biased and noisy
            # (std ~0.024 on this dataset), so a single-epoch peak there often
            # picks a fluke. Aggregating TP/FP/FN across the whole val set first
            # gives a stable, unbiased estimator. Background is excluded so
            # selection isn't dominated by the 64%-of-pixels easy class.
            bg_offset = 1 if 0 in self.class_values else 0
            val_iou_corpus_fg = float(np.nanmean(per_class_iou[bg_offset:]))

            # === LOGGING ===
            self.train_logs = {
                self.loss.__name__: train_loss_sum / len(self.train_loader),
                **{name: total / len(self.train_loader) for name, total in train_metric_sums.items()}
            }
            self.valid_logs = {
                self.loss.__name__: val_loss_sum / len(self.valid_loader),
                **{name: total / len(self.valid_loader) for name, total in val_metric_sums.items()}
            }

            is_best = val_iou_corpus_fg > max_score
            if is_best:
                print(f'New top score (corpus fg IoU): {val_iou_corpus_fg:.4f} > {max_score:.4f}')
                max_score = val_iou_corpus_fg
                epochs_without_improvement = 0
                self.save_model(epoch=epoch)
                self.save_loss_plots()
                if self.save_val_uncertainty:
                    self.save_uncertainty_images(epoch)
            else:
                epochs_without_improvement += 1

            if self.save_confusion_each_epoch or is_best:
                self.save_confusion_matrix(epoch=epoch)
            self.train_log_df.loc[epoch] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            self.valid_log_df.loc[epoch] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score'], val_iou_corpus_fg]
            self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
            self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_iou_corpus_fg)
            else:
                self.scheduler.step()

            sampled_dist = {}
            if self.sampler is not None:
                sampled_counts = self.train_patch_counts[self.sampler.last_indices].sum(axis=0)
                sampled_total = sampled_counts.sum()
                sampled_dist = {
                    f"sampled_dist/{label}": sampled_counts[i] / sampled_total if sampled_total > 0 else 0
                    for i, label in enumerate(self.labels)
                }

            per_class_iou_log = {
                f"val/iou_{label}": float(per_class_iou[i])
                for i, label in enumerate(self.labels)
            }
            print("Per-class val IoU: " + ", ".join(
                f"{label}={per_class_iou[i]:.3f}" for i, label in enumerate(self.labels)
            ))

            wandb_log = {
                "epoch": epoch,
                "train/loss": self.train_logs[self.loss.__name__],
                "train/iou": self.train_logs["iou_score"],
                "val/loss": self.valid_logs[self.loss.__name__],
                "val/iou": self.valid_logs["iou_score"],
                "val/iou_corpus_fg": val_iou_corpus_fg,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                **sampled_dist,
                **per_class_iou_log,
            }
            if 'iou_background' in self.valid_logs:
                wandb_log["train/iou_background"] = self.train_logs['iou_background']
                wandb_log["val/iou_background"] = self.valid_logs['iou_background']
            if hs_stats is not None:
                wandb_log.update(hs_stats)
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
        self.valid_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score', 'iou_corpus_fg'])


        # memory_optimized's gradient checkpointing is applied in prepare_model;
        # training itself uses the standard accumulation loop (the old separate
        # run_optimized_train_loop no longer exists after the dev-file merge).
        if self.accumulation_steps == 0:
            if self.use_amp:
                raise ValueError("--amp requires the gradient-accumulation "
                                 "training loop (accumulation_steps > 0).")
            self.run_train_loop()
        else:
            self.run_train_loop_with_accumulation(early_stopping_patience=early_stopping_patience)
        self.save_loss_plots()




        self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
        self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))


    def save_confusion_matrix(self, epoch, tile_top_per_pair=20, tile_min_pixels=50):
        """Accumulate the full-corpus confusion matrix over the validation set
        and (optionally) emit a per-tile diagnostic CSV listing the validation
        tiles that contribute the most pixels to each (true, pred) confusion
        pair — useful when a stuck confusion (e.g. cyanosbark→cyanosmoss) is
        suspected to be ground-truth labeling noise rather than model error.

        Args:
            epoch: integer epoch tag for output filenames.
            tile_top_per_pair: keep at most this many tiles per (true, pred)
                confusion pair in the diagnostic CSV. 0 disables the CSV.
            tile_min_pixels: confusions with fewer pixels than this in a single
                tile are not recorded (noise floor).
        """
        n_cls = len(self.class_values)
        cf_matrix = np.zeros((n_cls, n_cls), dtype=np.int64)
        val_to_label = {v: self.labels[i] for i, v in enumerate(self.class_values)}
        # Full ordered label list — passing this to sklearn.confusion_matrix
        # ensures every off-diagonal cell is counted, even confusions into
        # classes not present in this tile's ground truth. Previous code
        # filtered by labels-present-in-y_true, which silently dropped FPs
        # into absent classes and inflated per-class precision.
        all_labels = list(self.class_values)

        tile_records = [] if tile_top_per_pair > 0 else None

        for n in range(len(self.valid_dataset)):
            image, gt_mask = self.valid_dataset[n]
            pred = self.calculate_prediction(image)
            gt_mask_2d, _ = self.get_2d_image(gt_mask)

            y_pred_flat = pred.flatten()
            y_true_flat = gt_mask_2d.flatten()

            cf_matrix += confusion_matrix(y_true_flat, y_pred_flat, labels=all_labels)

            if tile_records is None:
                continue

            # Per-tile: for each true class present, count which non-target
            # classes its pixels got assigned to. One entry per (true, pred)
            # pair above the min_pixels noise floor.
            tile_name = Path(self.x_valid[n]).name
            for tv in np.unique(y_true_flat):
                if tv not in val_to_label:
                    continue
                true_mask = y_true_flat == tv
                true_total = int(true_mask.sum())
                pred_subset = y_pred_flat[true_mask]
                pred_vals, pred_counts = np.unique(pred_subset, return_counts=True)
                for pv, pc in zip(pred_vals, pred_counts):
                    if pv == tv or pv not in val_to_label or pc < tile_min_pixels:
                        continue
                    tile_records.append({
                        'tile': tile_name,
                        'true_class': val_to_label[tv],
                        'pred_class': val_to_label[pv],
                        'err_pixels': int(pc),
                        'true_pixels_in_tile': true_total,
                        'err_fraction': round(pc / true_total, 4) if true_total else 0.0,
                        'image_path': self.x_valid[n],
                        'mask_path': self.y_valid[n],
                    })

        cf_matrix_normed = normalize(cf_matrix, axis=1, norm='l1')
        disp = ConfusionMatrixDisplay(cf_matrix_normed, display_labels=self.labels)

        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        plt.colorbar(disp.im_, boundaries=np.linspace(0, 1, 11))
        plt.savefig(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.png".format(epoch)))
        plt.close(fig)

        cfm_df = pd.DataFrame(columns=[self.labels], index=self.labels, data=cf_matrix_normed)
        cfm_df.to_csv(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.csv".format(epoch)))

        # Raw (unnormalized) matrix preserves row/column totals so per-class IoU
        # (including background) can be recomputed offline: IoU_c = CM[c,c] /
        # (sum(CM[c,:]) + sum(CM[:,c]) - CM[c,c])
        cfm_raw_df = pd.DataFrame(columns=[self.labels], index=self.labels, data=cf_matrix)
        cfm_raw_df.to_csv(os.path.join(self.exp_dir, "confusion_matrix_raw_ep{0}.csv".format(epoch)))

        if tile_records:
            df = pd.DataFrame(tile_records)
            df = (df.sort_values(['true_class', 'pred_class', 'err_pixels'],
                                 ascending=[True, True, False])
                    .groupby(['true_class', 'pred_class'], group_keys=False)
                    .head(tile_top_per_pair)
                    .reset_index(drop=True))
            out_path = os.path.join(self.exp_dir, f"tile_confusions_ep{epoch}.csv")
            df.to_csv(out_path, index=False)
            print(f"Tile-level confusions saved: {out_path} "
                  f"({len(df)} rows, top {tile_top_per_pair} per pair, "
                  f"min {tile_min_pixels} px)")



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
        ax[1].plot(self.valid_log_df.index, self.valid_log_df['iou_score'], label='valid iou (per-batch avg)')
        if 'iou_corpus_fg' in self.valid_log_df.columns:
            ax[1].plot(self.valid_log_df.index, self.valid_log_df['iou_corpus_fg'],
                       label='valid iou (corpus fg, selection signal)', linestyle='--')
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
        print(self.model_save_path)
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
        # With EMA, all offline evaluation (confusion matrix, prediction,
        # entropy) uses the averaged weights — consistent with what was
        # validated and checkpointed.
        model = self.ema.module if getattr(self, 'ema', None) is not None else self.model
        model.eval()
        # model returns logits (activation=None); apply softmax here so
        # downstream code (.round(), entropy) sees probabilities.
        with torch.no_grad():
            pr_mask = torch.softmax(model(x_tensor), dim=1)

        return pr_mask

    def calculate_prediction(self, img, return_entropy=False):
        """calculate prediction for one image crop

        Uses argmax over softmax channels so every pixel is assigned to exactly
        one class, matching the in-loop IoU/CM accumulation. The previous
        softmax→.round() (threshold-0.5) approach silently dropped pixels whose
        peak probability was <0.5 (common for rare/ambiguous pixels in 8-class
        softmax) — those pixels defaulted to value 0, inflating background
        confusions in the CM and disagreeing with the in-loop selection signal.

        Args:
            img (np array): image read by opencv (color channels: BGR)

        Returns:
            np array: mask with class values (background still possible if it's
            a learned class — the model now actively predicts it).
        """

        pr_mask = self.get_model_output(img)  # (1, C, H, W) softmax probabilities
        probs = pr_mask.squeeze(0).cpu().numpy()  # (C, H, W)
        class_entropies = -np.sum(probs * np.log2(probs + 1e-10), axis=0)

        pred_idx = probs.argmax(axis=0)  # (H, W) channel index 0..C-1
        # Map channel index → class value (e.g. when ignore_background=True,
        # channel 0 may correspond to class value 1, etc.).
        class_values_arr = np.asarray(self.class_values, dtype=np.int64)
        out = class_values_arr[pred_idx]

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

        save_path = os.path.join(self.pred_output_dir, mask_name)

        # Paletted PNG: file stores categorical class indices (recoverable via
        # np.array(Image.open(...))), and the embedded palette maps each index
        # to its ontology hex colour for human-readable visualisation.
        im = Image.fromarray(mask.astype(np.uint8), mode="P")
        palette = [0] * (256 * 3)
        for entry in self.ontology["ontology"].values():
            v = entry["value"]
            hex_color = entry["color"].lstrip("#")
            palette[3 * v:3 * v + 3] = [
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            ]
        im.putpalette(palette)
        im.save(save_path)

    def save_entropy_image(self, entropies, img_name):

        if img_name.endswith(".JPG"):
            img_name = img_name.replace(".JPG", "_entropy.png")
        elif img_name.endswith(".jpg"):
            img_name = img_name.replace(".jpg", "_entropy.png")

        plt.figure()
        plt.imsave(os.path.join(self.pred_output_dir, img_name), entropies)
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
        padded_height = height + outer_gap + self.size - (height+outer_gap)%self.size
        padded_width = width + outer_gap + self.size - (width+outer_gap)%self.size

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

        self.load_model()

        self.pred_output_dir = Path(self.exp_dir) / Path(self.test_dir).name
        self.pred_output_dir.mkdir(parents=True, exist_ok=True)
        print("Writing predictions to:", self.pred_output_dir)

        img_paths = [os.path.join(self.test_dir, x) for x in os.listdir(self.test_dir) if x.lower().endswith(".jpg")]
        number_imgs = len(img_paths)
        n_ok, n_fail = 0, 0
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
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print("Error on {0}: {1}".format(img_paths[n], e))
                traceback.print_exc()
                if n_ok == 0 and n_fail >= 3:
                    raise RuntimeError(
                        "First {0} images all failed during prediction; aborting.".format(n_fail)
                    ) from e
                continue

        print("Predict done: {0} ok, {1} failed (out of {2})".format(n_ok, n_fail, number_imgs))

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
    parser.add_argument('--use_weighted_sampler', default=True, action=argparse.BooleanOptionalAction, help='use weighted sampler to address class imbalance in training data (disable with --no-use_weighted_sampler; the old "--use_weighted_sampler False" silently parsed as the truthy string "False")')
    parser.add_argument('--loss', default='focal', choices=['focal', 'dice_ce', 'tversky_ce'], help='training loss: "focal" (default), "dice_ce" (Dice + class-weighted CE), or "tversky_ce" (Tversky + class-weighted CE; tune alpha/beta to trade off FP vs FN on rare classes)', required=False)
    parser.add_argument('--class_weight_power', default=0.5, type=float, help='exponent p in CE class weight = 1/freq^p (dice_ce only). 0.5=sqrt (mild), 0.75=stronger rare-class push, 1.0=full inverse frequency (aggressive, can be unstable).', required=False)
    parser.add_argument('--bg_weight_multiplier', default=1.0, type=float, help='multiplies the background CE weight (dice_ce only). 1.0=default, 0.25=soft downweight, 0.0=hard ignore (also drops background from Dice).', required=False)
    parser.add_argument('--save_confusion_each_epoch', default=False, action='store_true', help='save confusion matrix every epoch (default: only on improvement). Useful for offline per-class IoU analysis at every epoch.')
    parser.add_argument('--tversky_alpha', default=0.3, type=float, help='Tversky FP weight (tversky_ce only). Lower alpha = less FP penalty.')
    parser.add_argument('--tversky_beta', default=0.7, type=float, help='Tversky FN weight (tversky_ce only). Higher beta = more FN penalty → higher recall on rare classes. alpha=beta=0.5 reduces to Dice.')
    parser.add_argument('--hard_sample_target', default=None, type=str, help='Enable hard sampling on this class (label name, e.g. "cyanosbark"). At end of each epoch, the per-tile FN rate for this class is EMA-smoothed and multiplied into the inv-freq tile weights so the next epoch oversamples tiles the model is currently failing on. Requires --use_weighted_sampler. Off by default.')
    parser.add_argument('--hard_sample_strength', default=2.0, type=float, help='Hard-sampling weight multiplier: new_w = base_w * (1 + strength * hardness_ema). 0 disables boost (= baseline), 2.0 doubles weight for max-hardness tiles, larger pushes harder.')
    parser.add_argument('--hard_sample_ema', default=0.7, type=float, help='EMA factor for per-tile hardness: hardness = ema*prev + (1-ema)*current. Higher = smoother (slow to react), lower = more reactive to current epoch.')
    # --- Optimizer / schedule / regularization (defaults reproduce legacy behavior) ---
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'], help='"adam" (legacy default, coupled L2 decay) or "adamw" (decoupled decay, no decay on biases/norms, supports --decoder_lr_mult; recommended for mit_* encoders).')
    parser.add_argument('--lr', default=1e-4, type=float, help='base (encoder) learning rate. Legacy default 1e-4; for adamw + mit_b5 try 6e-5.')
    parser.add_argument('--decoder_lr_mult', default=1.0, type=float, help='decoder/head LR = lr * this (adamw only). The randomly-initialised decoder tolerates a higher LR than the pretrained encoder; try 10.')
    parser.add_argument('--lr_schedule', default='plateau', choices=['plateau', 'cosine'], help='"plateau" (legacy ReduceLROnPlateau on val IoU) or "cosine" (linear warmup + cosine decay over --epochs; uses the budget better for short runs).')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='linear warmup epochs before cosine decay (cosine schedule only).')
    parser.add_argument('--amp', default=False, action='store_true', help='bf16 autocast mixed precision (Ampere+ GPUs): ~2x speed and ~half activation memory, enabling a larger physical batch (better BatchNorm statistics). Off by default.')
    parser.add_argument('--grad_checkpoint', default=False, action='store_true', help='gradient checkpointing for mit_* encoders WITHOUT the num_workers=0 side effect of --memory_optimized. Required for batch_size >= 4 at 1024^2 on a 40GB A100: attention softmax stays fp32 under autocast, so --amp alone does not fit batch 4. ~30%% slower per step, roughly offset by --amp.')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='CE label smoothing (dice_ce/tversky_ce). 0.05-0.1 makes CE tolerant of mislabeled pixels (e.g. cyanosbark/cyanosmoss ambiguity). 0 = legacy behavior.')
    parser.add_argument('--ema_decay', default=0.0, type=float, help='exponential moving average of model weights, updated each optimizer step (try 0.999). Validation, selection and best_model.pth use the EMA weights. 0 = off (legacy).')
    # --- Augmentation (defaults reproduce the long-standing hardcoded pipeline) ---
    parser.add_argument('--aug_photometric', default=False, action='store_true', help='add mild brightness/contrast + gentle hue/sat jitter to training augmentation. Off by default (legacy pipeline was geometry-only).')
    parser.add_argument('--aug_scale_limit', default=0.1, type=float, help='ShiftScaleRotate scale_limit in training augmentation (legacy default 0.1; try 0.25 for more scale variety).')
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
    loss_name = args.loss
    class_weight_power = args.class_weight_power
    bg_weight_multiplier = args.bg_weight_multiplier
    save_confusion_each_epoch = args.save_confusion_each_epoch
    tversky_alpha = args.tversky_alpha
    tversky_beta = args.tversky_beta
    hard_sample_target = args.hard_sample_target
    hard_sample_strength = args.hard_sample_strength
    hard_sample_ema = args.hard_sample_ema
    optimizer_name = {'adam': 'Adam', 'adamw': 'AdamW', 'sgd': 'SGD'}[args.optimizer]

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
                          num_workers=num_workers,
                          loss_name=loss_name,
                          class_weight_power=class_weight_power,
                          bg_weight_multiplier=bg_weight_multiplier,
                          save_confusion_each_epoch=save_confusion_each_epoch,
                          tversky_alpha=tversky_alpha,
                          tversky_beta=tversky_beta,
                          hard_sample_target=hard_sample_target,
                          hard_sample_strength=hard_sample_strength,
                          hard_sample_ema=hard_sample_ema,
                          optimizer_name=optimizer_name,
                          lr=args.lr,
                          decoder_lr_mult=args.decoder_lr_mult,
                          lr_schedule=args.lr_schedule,
                          warmup_epochs=args.warmup_epochs,
                          use_amp=args.amp,
                          grad_checkpoint=args.grad_checkpoint,
                          label_smoothing=args.label_smoothing,
                          ema_decay=args.ema_decay,
                          aug_photometric=args.aug_photometric,
                          aug_scale_limit=args.aug_scale_limit,
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

