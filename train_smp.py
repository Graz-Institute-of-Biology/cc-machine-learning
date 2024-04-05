import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import keras.backend as K
# K.set_image_data_format('channels_first')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from smp_dataset import Dataset
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



class Trainer():
    """Trainer class for training and testing segmentation model
    Used for manual training on GPU server, for model testing/evaluation and
    for predicting images
    """

    def __init__(self, encoder='efficientnet-b3', encoder_weights='imagenet', seed=10, load_config=True, device='cuda', size=1024, pred=False):

        self.size = size
        self.pred = pred
        if load_config:
            self.load_config()
            self.seed = seed
            # PATHS SERVER
            self.img_dir = Path(os.path.join(self.yaml_file["img_dir"], str(self.yaml_file["train_img"])))
            self.mask_dir = Path(os.path.join(self.yaml_file["mask_dir"], str(self.yaml_file["train_mask"])))
            self.exp_dir = Path(os.path.join(self.yaml_file["exp_dir"], "exp_{0}_{1}".format(encoder, self.seed)))
            self.test_dir = Path(os.path.join(self.yaml_file["test_dir"]))

            # image size: 2048, 1024, 512, 256
            self.size = self.yaml_file["size"]

        # model settings
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.class_dict = {	    "background" : 0,
                                "liverwort" : 1,
								"moss" : 2,
								"cyanosliverwort" : 3,
								"cyanosmoss" : 4,
								"lichen" : 5,
								"barkdominated" : 6,
								"cyanosbark" : 7,
								"other" : 8,
                            }
        self.ontology = {	    "background" : "#000000",
                                "liverwort" : "#1cffbb",
								"moss" : "#00bcff",
								"cyanosliverwort" : "#0059ff",
								"cyanosmoss" : "#2601d8",
								"lichen" : "#ff00c3",
								"barkdominated" : "#Ff0000",
								"cyanosbark" : "#FFA500",
								"other" : "#FFFF00",
                            }
        
        self.class_values = list(self.class_dict.values())
        
        self.activation = 'softmax2d'
        self.device = device
        self.lr_count = 0
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.lr_schedule = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
        self.lr = self.lr_schedule[self.lr_count]

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.preprocessing = get_preprocessing(self.preprocessing_fn)

    def load_config(self):
        """ load server config by default, if not available load local config
        """
        with open("server_config.yaml") as f:
            self.yaml_file = yaml.safe_load(f)
            self.best_model_path = self.yaml_file["best_model_path"]
            print(self.best_model_path)

        if not Path(self.yaml_file["img_dir"]).exists():
            with open("local_config.yaml") as f:
                self.yaml_file = yaml.safe_load(f)
                self.best_model_path = self.yaml_file["best_model_path"]

    def set_paths(self, train_split=0.8, train=False, test=False, pred=False):
        """Set train, valid, test paths for images and masks.
        """

        if train:
            # check if exp_dir exists and create new one
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

        self.ids = os.listdir(self.img_dir)
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

        self.focal_loss = smp.losses.FocalLoss(mode='multilabel', ignore_index=[0])
        self.focal_loss.__name__ = 'Focal_loss'
        self.loss = self.focal_loss

        self.metrics = [
            smp_metrics.IoU(threshold=0.5),
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
        ckpt_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'train_dl': self.train_loader,
            # 'valid_dl': self.valid_loader,
            'epoch': epoch,
            'ontology' : self.ontology,
        }
        if not update_latest:
            torch.save(ckpt_dict, self.model_save_path)
        else:
            save_path = os.path.join(self.exp_dir, 'latest_model.pth')
            torch.save(ckpt_dict, save_path)
        
        print("Model updated!")

    def train_model(self, epochs=1, save_every=10):

        print("Training on {0} images".format(len(self.images_fps)))
        max_score = 0

        self.train_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])
        self.valid_log_df = pd.DataFrame(columns=[self.loss.__name__, 'iou_score'])


        for i in range(0, epochs):
            
            print('\n Seed {0} | Epoch: {1}/{2}'.format(self.seed, i, epochs))
            self.train_logs = self.train_epoch.run(self.train_loader)
            self.valid_logs = self.valid_epoch.run(self.valid_loader)
            
            # do something (save model, change lr, etc.)
            if max_score < self.valid_logs['iou_score']:
                print('New top score: {0} > {1} '.format(self.valid_logs['iou_score'], max_score))
                max_score = self.valid_logs['iou_score']
                # torch.save(self.model, self.model_save_path)
                self.save_model(epoch=i)
                self.save_loss_plots()

            if (i+1)%save_every == 0:
                self.save_model(epoch=i, update_latest=True)
                self.save_loss_plots()
                
            # if (i+1)%25 == 0:
                # self.lr_count += 1
                # self.lr = self.lr_schedule[self.lr_count]
                # self.optimizer.param_groups[0]['lr'] = self.lr
                # print('Decrease decoder learning rate to {0}!'.format(self.lr))


            
            self.train_log_df.loc[i] = [self.train_logs[self.loss.__name__], self.train_logs['iou_score']]
            self.valid_log_df.loc[i] = [self.valid_logs[self.loss.__name__], self.valid_logs['iou_score']]


        self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
        self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))

        self.save_loss_plots()

    def save_loss_plots(self):
        """save plots for training and validation loss
        """
        self.pdf = PdfPages(self.pdf_path)

        title = "Model: {0}, Seed: {1}".format(self.encoder, self.seed)
        fig, ax = plt.subplots(figsize=(16, 5), ncols=2)
        fig.suptitle(title)

        ax[0].plot(self.train_log_df.index, self.train_log_df[self.loss.__name__], label='train Dice loss')
        ax[0].plot(self.valid_log_df.index, self.valid_log_df[self.loss.__name__], label='valid Dice loss')

        ax[1].plot(self.train_log_df.index, self.train_log_df['iou_score'], label='train iou score')
        ax[1].plot(self.valid_log_df.index, self.valid_log_df['iou_score'], label='valid iou score')
        plt.show()
        plt.legend()

        self.pdf.savefig(fig)
        self.pdf.close()
        
    def save_images_to_pdf(self, title=None, **images):
        """
        create plots for training and validation loss,
        visualize images in a row: input image, ground truth and prediction
        """
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.patches as mpatches

        n = len(images)

        labels = list(self.class_dict.keys())

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
                image = image.transpose(1, 2, 0)
                out = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

                for v in self.class_values:
                    out[image[:,:,v] == 1] = v
                image = out
                unique_values = np.unique(image)

                plt.imshow(image, cmap=col, interpolation='nearest', norm=norm)
                # patches = [ mpatches.Patch(color=colors_hex[i], label=labels[i] ) for i in range(len(colors))]

                # plot all class values in legend
                # patches = [ mpatches.Patch(color=colors_hex[e], label=labels[i] ) for i,e in enumerate(self.class_values)]
                
                # plot only unique class values in legend
                patches = [ mpatches.Patch(color=colors_hex[i], label=labels[i] ) for i in unique_values]

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

    def calculate_prediction(self, img):
        """calculate prediction for one image crop

        Args:
            img (np array): image read by opencv (color channels: BGR)

        Returns:
            np array: mask array with class values, color coded with class colors
        """

        x_tensor = torch.from_numpy(img).to(self.device).unsqueeze(0)
        self.model.eval()
        pr_mask = self.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = pr_mask.transpose(1, 2, 0)

        out = np.zeros((pr_mask.shape[0], pr_mask.shape[1]), dtype=np.int64)

        for v in self.class_values:
            out[pr_mask[:,:,v] == 1] = v

        return out
    
    def save_output_mask(self, mask, img_name):
        from PIL import Image

        save_path = os.path.join(self.test_dir, img_name.replace(".JPG", "_mask_04.png"))
        im = Image.fromarray(mask)
        im.save(save_path)

    def predict_whole_image(self, img, debug=False):
        """predict whole image by dividing it into 1024x1024 crops and calculate prediction for each crop
            if debug is True, return empty mask for testing

        Args:
            img (np.ndarray or PIL): _description_

        Returns:
            np.ndarray: mask results for whole image (categorical values)
        """
        
        print("predicting...")
        # create zeros array that is divisible by 1024
        if not type(img) == np.ndarray:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # size = 1024
        original_img = img
        width = int(np.ceil(img.shape[1]/self.size) * self.size)
        height = int(np.ceil(img.shape[0]/self.size) * self.size)
        padded_img = np.zeros((height, width, 3), dtype=np.uint8)
        padded_img[:img.shape[0], :img.shape[1], :] = img

        whole_mask = np.zeros((height, width), dtype=np.uint8)
        width_steps = np.arange(int(width/self.size))
        height_steps = np.arange(int(height/self.size))

        if debug:
            return whole_mask


        for w in tqdm(width_steps):
            for h in height_steps:
                img = padded_img[h*self.size:(h+1)*self.size, w*self.size:(w+1)*self.size, :]
                img = self.preprocessing(image=img)['image']

                color_coded_mask = self.calculate_prediction(img)
                whole_mask[h*self.size:(h+1)*self.size, w*self.size:(w+1)*self.size] = color_coded_mask
                

                # plt.figure()
                # plt.imshow(img)
                # plt.show()
        whole_mask = whole_mask[:original_img.shape[0], :original_img.shape[1]]

        return whole_mask
    
    def predict(self, load_model=True, test_dir=None, model_path=None):
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
            print("{0} / {1} | ".format(n, number_imgs, img_paths[n]))

            img = cv2.imread(img_paths[n])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            whole_mask = self.predict_whole_image(img)
            self.save_output_mask(whole_mask, img_name)
            

def parse_args():
    parser = argparse.ArgumentParser(description='Train or test segmentation model')
    parser.add_argument('--mode', default='predict', type=str, help='mode: "train" for training model \n "test" for testing with ground truth and save results to pdf file \n "predict" for predicting whole images and save prediction mask', required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # encoder_list = ['mit_b0', 'efficientnet-b3', 'efficientnet-b7', 'vgg16', 'resnet50']
    encoder_list = ['mit_b3']
    # seeds = [10, 20, 30, 40, 50]
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    # seeds = [10]
    
    args = parse_args()

    if args.mode == 'train':
        for encoder in encoder_list:
            for seed in seeds:
                print("Train: ", encoder, "\n")
                print("Seed: ", seed, "\n")
                trainer = Trainer(encoder=encoder, seed=seed) # create Trainer object and set default values

                trainer.set_paths(train_split=0.9, train=True) # set paths for training
                trainer.create_dataloaders(augmentations=True) # create dataloaders from image and mask paths
                trainer.prepare_model() # create Unet object and loss & metric objects and Training/Validation Epoch Runners
                trainer.train_model(epochs=2000) # start training routine using Training/Validation Epoch Runners
                trainer.test_model()
    
    elif args.mode == 'test':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Test: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seeds[0]) # create Trainer object and set default values
        trainer.set_paths() # set paths for testing (not recently tested)
        trainer.test_model() # test model

    elif args.mode == 'predict':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Predict: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, seed=seeds[0], pred=True)
        trainer.set_paths(pred=True)
        trainer.predict()

