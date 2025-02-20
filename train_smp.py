import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import keras.backend as K
# K.set_image_data_format('channels_first')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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


class Trainer():
    """Trainer class for training and testing segmentation model
    Used for manual training on GPU server, for model testing/evaluation and
    for predicting images
    """

    def __init__(self, 
                 encoder='efficientnet-b3', 
                 encoder_weights='imagenet', 
                 seed=10,
                 load_config=True,
                 device='cuda', 
                 size=1024, 
                 pred=False, 
                 active_learning=False, 
                 save_val_uncertainty=False,
                 config_file="None", ):

        self.size = size
        self.pred = pred
        self.seed = seed
        self.active_learning=active_learning
        self.save_val_uncertainty = save_val_uncertainty
        self.ignore_background = True
        self.config_file = config_file
        self.device = device

        if self.active_learning:
            self.save_val_uncertainty = True
        
        if load_config:
            self.load_config()


        # model settings
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.load_ontology()
        
        self.class_values = [d["value"] for d in self.ontology["ontology"].values()]
        self.labels = list(self.ontology["ontology"].keys())
        
        self.activation = 'softmax2d'
        self.lr_count = 0
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.lr_schedule = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
        self.lr = self.lr_schedule[self.lr_count]
        self.uncertainy_runs = 10

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.preprocessing = get_preprocessing(self.preprocessing_fn)

    def load_ontology(self):
        """set ontologies, should be moved to separate file!!!
        """
        ontology_file_name = "ontology_{0}.json".format(self.ontology_name)
        
        with open(ontology_file_name) as f:
            self.ontology = json.load(f)
            print(self.ontology)



        # if self.ontology_name == "atto":
        #     self.class_dict = {	    "background" : 0,
        #                             "liverwort" : 1,
        #                             "moss" : 2,
        #                             "cyanosliverwort" : 3,
        #                             "cyanosmoss" : 4,
        #                             "lichen" : 5,
        #                             "barkdominated" : 6,
        #                             "cyanosbark" : 7,
        #                             "other" : 8,
        #                         }
        #     self.ontology = {	    "background" : "#000000",
        #                             "liverwort" : "#1cffbb",
        #                             "moss" : "#00bcff",
        #                             "cyanosliverwort" : "#0059ff",
        #                             "cyanosmoss" : "#2601d8",
        #                             "lichen" : "#ff00c3",
        #                             "barkdominated" : "#Ff0000",
        #                             "cyanosbark" : "#FFA500",
        #                             "other" : "#FFFF00",
        #                         }
            
        # elif self.ontology_name == "fbground":
        #     self.class_dict = {
        #                 "background" : 0,	
        #                 "foreground" : 1,}
            
        #     self.ontology = {
        #                 "background" : "#000000",	
        #                 "foreground" : "#FFFF00",}
            
        #     self.ignore_background = False
            
        # elif self.ontology_name == "gg":
        #     self.class_dict = {
        #                 "background" : 0,	
        #                 "cyano_-_dominated" : 1,
        #                 "lichen" : 2,
        #                 "moss" : 3,
        #                 "vascular plants" : 4,
        #                 "rock" : 5,
        #                 "other" : 6,
        #                 "fungi" : 7,}
            
        #     self.ontology = {
        #                 "background" : "#000000",	
        #                 "cyano_-_dominated" : "#1CE6FF",
        #                 "lichen" : "#ffdb0c",
        #                 "moss" : "#FF4A46",
        #                 "vascular plants" : "#008941",
        #                 "rock" : "#dc22e6",
        #                 "other" : "#B79762",
        #                 "fungi" : "#7A4900",
        #                 "markers" : "#FFDBE5",}
            
        # elif self.ontology_name == "graz":
        #     self.class_dict = {
		# 							"background" : 0,
		# 							"bryophyte" : 1,
		# 							"lichen" : 2,
		# 							"barkdominated" : 3,
		# 							"other" : 4,}
            
        #     self.ontology = {
        #                     "background" : "#000000",
        #                     "bryophyte" : "#1CE6FF",
        #                     "lichen" : "#FF34FF",
        #                     "barkdominated" : "#FF4A46",
        #                     "other" : "#928e00",}

            
            
            
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

    def load_config(self):
        """ load server config by default, if not available load local config
        """

        if self.config_file == "None":
            self.config_file = "server_config.yaml"

        with open(self.config_file) as f:
            self.yaml_file = yaml.safe_load(f)
            self.best_model_path = self.yaml_file["best_model_path"]
            self.ontology_name = self.yaml_file["ontology"]
            self.use_gpu = self.yaml_file["use_gpu"]
            if not self.use_gpu:
                self.device = 'cpu'
                print("Local testing? \n Using CPU! \n GPU NOT ACTIVATED!!!")
            print(self.best_model_path)

        # load local config if path does not exist
        if not Path(self.yaml_file["img_dir"]).exists():
            print("IMG Path does not exist\n Fall back to local config!")
            with open("local_config.yaml") as f:
                self.yaml_file = yaml.safe_load(f)
                self.best_model_path = self.yaml_file["best_model_path"]


        # PATHS SERVER
        self.img_dir = Path(os.path.join(self.yaml_file["img_dir"], str(self.yaml_file["train_img"])))
        self.mask_dir = Path(os.path.join(self.yaml_file["mask_dir"], str(self.yaml_file["train_mask"])))
        self.test_dir = Path(os.path.join(self.yaml_file["test_dir"]))

        # image size: 2048, 1024, 512, 256
        self.size = self.yaml_file["size"]
        self.forest_type = self.yaml_file["train_img"][0]


    def set_non_unique_paths(self, train_split):
        self.images_fps = sorted([os.path.join(self.img_dir, image_id) for image_id in self.ids if image_id.lower().endswith(".jpg")])
        self.masks_fps = sorted([os.path.join(self.mask_dir, image_id.replace("JPG", "png")) for image_id in self.ids])


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

    def set_unique_paths(self, train_split, unique_imgs):
        shuffled_original = unique_imgs.copy()
        np.random.shuffle(shuffled_original)

        if self.active_learning:
            train_split = 0.1
            train_len_ = int(len(unique_imgs) * train_split)
            val_len_ = int(train_len_*2)
            unique_train_imgs = shuffled_original[:train_len_]
            unique_val_imgs = shuffled_original[train_len_:train_len_ + val_len_]
            unique_pool_imgs = shuffled_original[val_len_:]
            self.unique_pool_imgs = unique_pool_imgs
        else:
            train_len_ = int(len(unique_imgs) * train_split)
            unique_train_imgs = shuffled_original[:train_len_]
            unique_val_imgs = shuffled_original[train_len_:]


        self.x_train = [os.path.join(self.img_dir, x) for x in self.ids if x.split("_part")[0] in unique_train_imgs]
        self.y_train = [os.path.join(self.mask_dir, x.replace("JPG", "png")) for x in self.ids if x.split("_part")[0] in unique_train_imgs]

        self.x_valid = [os.path.join(self.img_dir, x) for x in self.ids if x.split("_part")[0] in unique_val_imgs]
        self.y_valid = [os.path.join(self.mask_dir, x.replace("JPG", "png")) for x in self.ids if x.split("_part")[0] in unique_val_imgs]
        

        self.x_test = self.x_valid.copy()
        self.y_test = self.y_valid.copy()

        if self.active_learning:
            self.x_pool = [os.path.join(self.img_dir, x) for x in self.ids if x.split("_part")[0] in unique_pool_imgs]
            self.y_pool = [os.path.join(self.mask_dir, x.replace("JPG", "png")) for x in self.ids if x.split("_part")[0] in unique_pool_imgs]
            np.random.shuffle(self.x_test)
            self.x_test = self.x_test[:20]
            self.y_test = [x.replace("JPG", "png").replace("img", "mask") for x in self.x_test]

    def set_paths(self, train_split=0.8, train=False, test=False, pred=False):
        """Set train, valid, test paths for images and masks.
        """

        if train:
            # check if exp_dir exists and create new one
            self.exp_dir = Path(os.path.join(self.yaml_file["exp_dir"], "exp_{0}_{1}_{2}".format(self.forest_type, self.encoder, self.seed)))
            exp_dir = Path(str(self.exp_dir) + "_" + str(uuid.uuid4())[:6])
            while exp_dir.exists():
                exp_dir = Path(str(self.exp_dir) + "_" + str(uuid.uuid4())[:6])

            self.exp_dir = exp_dir
            os.makedirs(self.exp_dir)

        if test:
            self.exp_dirs = [x for x in os.listdir(Path(self.yaml_file["exp_dir"])) if encoder in x]
            self.sorted_exp_dirs = sorted(self.exp_dirs, key=lambda x: int(x.split("_")[-1]))
            self.exp_dir = Path(os.path.join(Path(self.yaml_file["exp_dir"]), self.sorted_exp_dirs[-1]))
            print(self.exp_dir)
        if pred:
            self.exp_dir = Path(os.path.join(self.yaml_file["exp_dir"], "exp_{0}_{1}".format(self.encoder, self.seed)))
            print("Predicting images from: ", self.test_dir)

        pdf_path = os.path.join(self.exp_dir, "results.pdf")
        count = 1
        while os.path.exists(pdf_path):
            pdf_path = os.path.join(self.exp_dir, "results_{0}.pdf".format(count))
            count += 1
            
        self.pdf_path = pdf_path

        # set model save path
        self.model_save_path = os.path.join(self.exp_dir, 'best_model.pth')
        np.random.seed(self.seed)

        self.ids = sorted(os.listdir(self.img_dir))

        # get unique image names
        original_imgs = [x.split("_part")[0] for x in self.ids]
        unique_imgs = sorted(np.unique(original_imgs))
        print(unique_imgs)
        if len(unique_imgs) == 0:
            print("Error: No images found!")
            return None

        if len(unique_imgs) < 10:
            print("Only {0} image found! Using non-unique images. Data leakage might be a problem...".format(len(unique_imgs)))
            self.set_non_unique_paths(train_split)

        elif len(unique_imgs) >= 10:
            print("Multiple images found! Using unique images to tackle data leakage")
            self.set_unique_paths(train_split, unique_imgs)



        


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

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=1)


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

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=1)

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


        #  self.dice_loss = smp.losses.DiceLoss(mode='multilabel', ignore_index=[0])
        # self.dice_loss.__name__ = 'Dice_loss'
        # self.loss = self.dice_loss


        
        if self.ignore_background:
            ignore_index = [0]
            ignore_channels = [0]
        else:
            ignore_index = None
            ignore_channels = None

        self.focal_loss = smp.losses.FocalLoss(mode='multilabel', ignore_index=ignore_index)
        self.focal_loss.__name__ = 'Focal_loss'
        self.loss = self.focal_loss

        self.metrics = [
            smp_metrics.IoU(threshold=0.5, ignore_channels=ignore_channels),
        ]

        self.optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=self.lr),
        ])



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
        }
        if not update_latest:
            torch.save(ckpt_dict, self.model_save_path)
        else:
            save_path = os.path.join(self.exp_dir, 'latest_model.pth')
            torch.save(ckpt_dict, save_path)
        
        print("Model updated!")

    def train_model(self, epochs=1, al_step=25):

        print("Training on {0} images".format(len(self.ids)))
        max_score = 0

        self.train_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])
        self.valid_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])

        if self.active_learning:
            # epochs = epochs//al_step
            print("Active Learning is enabled!")

        for epoch in range(0, epochs):
            
            print('\n Seed {0} | Epoch: {1}/{2}'.format(self.seed, epoch, epochs))
            self.train_logs = self.train_epoch.run(self.train_loader)
            self.valid_logs = self.valid_epoch.run(self.valid_loader)

            if self.active_learning and (epoch+1)%al_step == 0:
                train_img_sorted = self.calculate_uncertainty_scores()
                if not train_img_sorted == None:
                    self.update_dataloaders(train_img_sorted)
            
            # do something (save model, change lr, etc.)
            if max_score < self.valid_logs['iou_score']:
                print('New top score: {0} > {1} '.format(self.valid_logs['iou_score'], max_score))
                max_score = self.valid_logs['iou_score']
                # torch.save(self.model, self.model_save_path)
                self.save_model(epoch=epoch)
                self.save_loss_plots()
                self.save_confusion_matrix(epoch=epoch)

                if self.save_val_uncertainty:
                    self.save_uncertainty_images(epoch)


            # if (i+1)%25 == 0:
                # self.lr_count += 1
                # self.lr = self.lr_schedule[self.lr_count]
                # self.optimizer.param_groups[0]['lr'] = self.lr
                # print('Decrease decoder learning rate to {0}!'.format(self.lr))


            
            self.train_log_df.loc[epoch] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            self.valid_log_df.loc[epoch] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score']]


        self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
        self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))

        self.save_loss_plots()

    def save_confusion_matrix(self, epoch):
        cf_matrix = np.zeros((len(self.class_values), len(self.class_values)))

        for n in range(len(self.valid_dataset)):
            # n = np.random.choice(len(valid_dataset))
            # image_vis = valid_dataset[n][0].astype('uint8')
            image, gt_mask = self.valid_dataset[n]
            pred = self.calculate_prediction(image)
            gt_mask_2d, unique_values = self.get_2d_image(gt_mask)

            y_pred_flattened = pred.flatten()
            y_true_flattened = gt_mask_2d.flatten()

            cf_m = confusion_matrix(y_true_flattened, y_pred_flattened)
            for i in range(len(unique_values)):
                u = unique_values[i]
                for j in range(len(unique_values)):
                    k = unique_values[j]
                    cf_matrix[u][k] += cf_m[i][j]


        cf_matrix_normed = normalize(cf_matrix, axis=1, norm='l1')
        disp = ConfusionMatrixDisplay(cf_matrix_normed, display_labels=self.labels)

        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        plt.colorbar(disp.im_, boundaries=np.linspace(0, 1, 11))
        plt.savefig(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.png".format(epoch)))

        cfm_df = pd.DataFrame(columns=[self.labels], index=self.labels, data=cf_matrix_normed)
        cfm_df.to_csv(os.path.join(self.exp_dir, "confusion_matrix_ep{0}.csv".format(epoch)))



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
        ax[1].set_ylim(0.6, 1.0)
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

        for v in self.class_values:
            out[mask[:,:,v] == 1] = v
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

        colors_hex = ["#000000","#1cffbb", "#00bcff","#0059ff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
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
                # image = image.transpose(1, 2, 0)
                # out = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

                # for v in self.class_values:
                #     out[image[:,:,v] == 1] = v
                # image = out
                # unique_values = np.unique(image)

                image, unique_values = self.get_2d_image(image)

                plt.imshow(image, cmap=col, interpolation='nearest', norm=norm)
                # patches = [ mpatches.Patch(color=colors_hex[i], label=labels[i] ) for i in range(len(colors))]

                # plot all class values in legend
                # patches = [ mpatches.Patch(color=colors_hex[e], label=labels[i] ) for i,e in enumerate(self.class_values)]
                
                # plot only unique class values in legend
                patches = [ mpatches.Patch(color=colors_hex[i], label=self.labels[i] ) for i in unique_values]

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
            checkpoint = torch.load(self.model_save_path)
        elif self.pred:
            print(self.best_model_path)
            checkpoint = torch.load(self.best_model_path)
        print("Checkpoint loaded!")
        self.print_summary(checkpoint)
        self.prepare_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            pr_mask = self.model.predict(x_tensor)
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
        pr_mask = self.model.predict(x_tensor)

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

        for v in self.class_values:
            out[pr_mask[:,:,v] == 1] = v

        if return_entropy:
            return out, class_entropies
        else:
            return out
    
    def save_mask(self, mask, img_name):
        from PIL import Image

        save_path = os.path.join(self.test_dir, img_name.replace(".JPG", "_mask.png"))
        im = Image.fromarray(mask)
        im.save(save_path)
        
    def save_entropy_image(self, entropies, img_name):

        plt.figure()
        plt.imsave(os.path.join(self.test_dir, img_name.replace(".JPG", "_entropy.png")), entropies)
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
                print("Coords: ", h1, h2, w1, w2)
                print("IMAGE: ", sub_img.shape)

                if get_entropies:
                    color_coded_mask, entropies = self.calculate_prediction(sub_img, return_entropy=True)
                    entropies = entropies[outer_gap:self.size-outer_gap, outer_gap:self.size-outer_gap]
                    print("Entropies:", entropies.shape)
                    whole_entropies[h1+outer_gap:h2-outer_gap, w1+outer_gap:w2-outer_gap] = entropies
                else:
                    color_coded_mask = self.calculate_prediction(sub_img)
                
                color_coded_mask = color_coded_mask[outer_gap:focus_size + outer_gap, outer_gap:focus_size + outer_gap]
                print("Colored mask:", color_coded_mask.shape)
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
            

def parse_args():
    parser = argparse.ArgumentParser(description='Train or test segmentation model')
    parser.add_argument('--mode', default='train', type=str, help='mode: "train" for training model \n "test" for testing with ground truth and save results to pdf file \n "predict" for predicting whole images and save prediction mask', required=False)
    parser.add_argument('--encoders', action="append", type=str, help='list of encoders to train/test/predict', required=True)
    parser.add_argument('--config', default='server_local_debug.yml', type=str, help='config file for training/testing/predicting', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # config_file = 'server_config_graz.yml'

    # encoder_list = ['mit_b0', 'efficientnet-b3', 'efficientnet-b7', 'vgg16', 'resnet50']
    # encoder_list = ['mit_b1', 'mit_b3', 'mit_b5']

    seeds = [20]
    
    args = parse_args()

    encoder_list = args.encoders
    config_file = args.config

    print(config_file)

    if args.mode == 'train':
        trainer = Trainer(active_learning=False, save_val_uncertainty=True, config_file=config_file) # create Trainer object, load config & set default values
        for encoder in encoder_list:
            trainer.set_encoder(encoder) # set encoder for model
            for seed in seeds:
                print("Train: ", encoder, "\n")
                print("Seed: ", seed, "\n")
                trainer.set_seed(seed) # set seed for reproducibility

                trainer.set_paths(train_split=0.8, train=True) # set paths for training
                trainer.create_dataloaders(augmentations=True) # create dataloaders from image and mask paths
                trainer.prepare_model() # create Unet object and loss & metric objects and Training/Validation Epoch Runners
                trainer.train_model(epochs=400) # start training routine using Training/Validation Epoch Runners
                trainer.test_model()
    
    elif args.mode == 'test':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Test: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seeds[0], config_file=config_file) # create Trainer object and set default values
        trainer.set_paths() # set paths for testing (not recently tested)
        trainer.test_model() # test model

    elif args.mode == 'predict':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Predict: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seeds[0], pred=True, config_file=config_file)
        trainer.set_paths(pred=True)
        trainer.predict(save_entropies=True)

