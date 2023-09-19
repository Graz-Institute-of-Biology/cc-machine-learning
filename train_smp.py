import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import keras.backend as K
# K.set_image_data_format('channels_first')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from smp_dataset import Dataset
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics as smp_metrics
import ssl
import random
from pathlib import Path
import yaml
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from training_helper import get_preprocessing, get_training_augmentation, get_validation_augmentation
ssl._create_default_https_context = ssl._create_unverified_context



class Trainer():

    def __init__(self, encoder='efficientnet-b3', encoder_weights='imagenet', test=False, model_path=None):

        with open("config.yaml") as f:
            yaml_file = yaml.safe_load(f)

        # PATHS SERVER
        self.img_dir = Path(os.path.join(yaml_file["img_dir"], str(yaml_file["size"])))
        self.mask_dir = Path(os.path.join(yaml_file["mask_dir"], str(yaml_file["size"])))
        self.exp_dir = Path(os.path.join(yaml_file["exp_dir"], "exp_{0}".format(encoder)))
        self.test_dir = Path(os.path.join(yaml_file["test_dir"]))
        self.best_model_path = model_path

        if not test:
            # check if exp_dir exists and create new one
            count = 1
            exp_dir = self.exp_dir
            while exp_dir.exists():
                exp_dir = Path(str(self.exp_dir) + "_{0}".format(count))
                count += 1

            self.exp_dir = exp_dir
            os.makedirs(self.exp_dir)

        elif test:
            self.exp_dirs = sorted([x for x in os.listdir(Path(yaml_file["exp_dir"])) if encoder in x])
            print(self.exp_dirs)
            self.exp_dir = Path(os.path.join(Path(yaml_file["exp_dir"]), self.exp_dirs[-1]))
            print(self.exp_dir)

        pdf_path = os.path.join(self.exp_dir, "results.pdf")
        count = 1
        while os.path.exists(pdf_path):
            pdf_path = os.path.join(self.exp_dir, "results_{0}.pdf".format(count))
            count += 1
            
        self.pdf_path = pdf_path

        # set model save path
        self.model_save_path = os.path.join(self.exp_dir, 'best_model.pth')

        # image size: 2048, 1024, 512, 256
        self.size = yaml_file["size"]

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
        
        self.class_values = list(self.class_dict.values())
        
        self.activation = 'softmax2d'
        self.device = 'cuda'
        self.lr = 0.0001

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)


    def set_paths(self, train_split=0.8, seed=10):
        """Set train, valid, test paths for images and masks.
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.ids = os.listdir(self.img_dir)
        self.images_fps = sorted([os.path.join(self.img_dir, image_id) for image_id in self.ids])
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


    def create_dataloaders(self):
        self.train_dataset = Dataset(
            self.x_train, 
            self.y_train,
            class_values=self.class_values,
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        self.valid_dataset = Dataset(
            self.x_valid, 
            self.y_valid,
            class_values=self.class_values,
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=1)



    def prepare_model(self):
        """Prepare model for training. Set loss, metrics, optimizer and training / validation runners.
        """

        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name=self.encoder, 
            encoder_weights=self.encoder_weights, 
            classes=len(self.class_values), 
            activation=self.activation,
        )


        self.loss = smp.losses.DiceLoss(mode='multilabel')
        self.loss.__name__ = 'Dice_loss'

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


    def train_model(self, epochs=1):

        print("Training on {0} images".format(len(self.images_fps)))
        max_score = 0

        self.train_log_df = pd.DataFrame(columns=['Dice_loss', 'iou_score'])
        self.valid_log_df = pd.DataFrame(columns=['Dice_loss', 'iou_score'])


        for i in range(0, epochs):
            
            print('\nEpoch: {}'.format(i))
            self.train_logs = self.train_epoch.run(self.train_loader)
            self.valid_logs = self.valid_epoch.run(self.valid_loader)
            
            # do something (save model, change lr, etc.)
            if max_score < self.valid_logs['iou_score']:
                max_score = self.valid_logs['iou_score']
                torch.save(self.model, self.model_save_path)
                print('Model saved!')
                
            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
            
            self.train_log_df.loc[i] = [self.train_logs['Dice_loss'], self.train_logs['iou_score']]
            self.valid_log_df.loc[i] = [self.valid_logs['Dice_loss'], self.valid_logs['iou_score']]


        self.train_log_df.to_csv(os.path.join(self.exp_dir, "train_log.csv"))
        self.valid_log_df.to_csv(os.path.join(self.exp_dir, "valid_log.csv"))

        self.pdf = PdfPages(self.pdf_path)
        self.save_loss_plots()

    def save_loss_plots(self):
        title = "Model: {0}, Seed: {1}".format(self.encoder, self.seed)
        fig, ax = plt.subplots(figsize=(16, 5), ncols=2)
        fig.suptitle(title)

        ax[0].plot(self.train_log_df.index, self.train_log_df['Dice_loss'], label='train Dice loss')
        ax[0].plot(self.valid_log_df.index, self.valid_log_df['Dice_loss'], label='valid Dice loss')

        ax[1].plot(self.train_log_df.index, self.train_log_df['iou_score'], label='train iou score')
        ax[1].plot(self.valid_log_df.index, self.valid_log_df['iou_score'], label='valid iou score')
        plt.show()
        plt.legend()

        self.pdf.savefig(fig)
        
    def save_images_to_pdf(self, title=None, **images):
        """
        create plots for training and validation loss,
        visualize images in a row: input image, ground truth and prediction
        """
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.patches as mpatches
        from PIL import ImageColor

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
            print(name)
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

                print(colors_hex)
                print(labels)
                print(unique_values)
                
                # plot only unique class values in legend
                patches = [ mpatches.Patch(color=colors_hex[i], label=labels[i] ) for i in unique_values]

                plt.axis('off')
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0. )   
            else:
                plt.imshow(image)
        # plt.show()
        self.pdf.savefig(fig)
        plt.close()

    def load_model(self):
        """load model from model_save_path
        """
        print("Loading model from: ")
        print(self.model_save_path)
        if not self.best_model_path == None:
            self.model = torch.load(self.best_model_path)
            self.exp_dir = Path(self.best_model_path).parent
            print("Set exp dir to: ", self.exp_dir)
        else:
            self.model = torch.load(self.model_save_path)
        print("Model loaded!")

    def set_pdf_path(self):
        """set path to pdf file for saving predictions and results
        """
        pdf_path = os.path.join(self.exp_dir, "predictions.pdf")

        count = 1
        while os.path.exists(pdf_path):
            pdf_path = os.path.join(self.exp_dir, "predictions_{0}.pdf".format(count))
            count += 1
        self.pdf = PdfPages(pdf_path)

    def test_model(self):
        valid_dataset = Dataset(
            self.x_test, 
            self.y_test,
            class_values=self.class_values,
            preprocessing=get_preprocessing(self.preprocessing_fn),
        )

        self.class_dict = {"Background" : 0,
                        "liverwort" : 1,
                        "Bryophytes" : 2,
                        "cyanosliverwort" : 3,
                        "Bryophytes & Cyanos" : 4,
                        "Lichen" : 5,
                        "Bark-Dominated" : 6,
                        "Bark & Cyanos" : 7,
                        "Other" : 8,
                    }
        
        self.class_values = list(self.class_dict.values())

        self.set_pdf_path()
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

    def predict(self):
        """ calculate predictions for all images in test folder and save it to pdf
        """

        self.set_pdf_path()
        self.load_model()

        img_paths = [os.path.join(self.test_dir, x) for x in os.listdir(self.test_dir)]
        preprocessing = get_preprocessing(self.preprocessing_fn)

        img_paths = random.sample(img_paths, 35)
        for n in range(len(img_paths)):

            img_name = Path(img_paths[n]).name
            print(img_name)
            img = cv2.imread(img_paths[n])
            try:
                img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                break
            img = preprocessing(image=img_show)['image']
        
            x_tensor = torch.from_numpy(img).to(self.device).unsqueeze(0)
            # img = img.transpose(1, 2, 0)
            pr_mask = self.model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
            self.save_images_to_pdf(
                title=img_name,
                image=img_show, 
                predicted_mask=pr_mask
            )

        self.pdf.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train or test segmentation model')
    parser.add_argument('--mode', default='train', type=str, help='mode: train or test')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # encoder_list = ['mit_b0', 'efficientnet-b3', 'efficientnet-b7', 'vgg16', 'resnet50']
    encoder_list = ['mit_b5']
    # seeds = [20,30,40,50]
    seeds = [60, 70, 80, 90, 100, 110]

    args = parse_args()

    if args.mode == 'train':
        for encoder in encoder_list:
            for seed in seeds:
                print("Train: ", encoder, "\n")
                print("Seed: ", seed, "\n")
                trainer = Trainer(encoder=encoder)

                trainer.set_paths(seed=seed)
                trainer.create_dataloaders()
                trainer.prepare_model()
                trainer.train_model(epochs=500)
                trainer.test_model()
    
    elif args.mode == 'test':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        print("Test: ", encoder, "\n")
        model_path = "/usr/people/EDVZ/faulhamm/cc-machine-learning/experiments/exp_mit_b5_1/best_model.pth"
        trainer = Trainer(encoder=encoder, test=True, model_path=model_path)
        trainer.set_paths()
        trainer.test_model()

    elif args.mode == 'predict':
        # for encoder in encoder_list:
        encoder = 'mit_b5'
        model_path = "/usr/people/EDVZ/faulhamm/cc-machine-learning/experiments/exp_mit_b5_1/best_model.pth"
        print("Predict: ", encoder, "\n")
        trainer = Trainer(encoder=encoder, test=True, model_path=model_path)
        trainer.set_paths()
        trainer.predict()
