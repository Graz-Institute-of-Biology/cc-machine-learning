import os
import pickle
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
import pathlib
from customdatasets import SegmentationDataSet
from transformations import Compose, DenseTarget, RandomFlip, Resize_Sample
from transformations import MoveAxis, Normalize01, RandomCrop, RandomCropVal_JEM, RandomCropTrain_JEM
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from os import walk
import torch as t
import numpy as np
import torch.nn as nn


def get_files(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for names in filenames:
            files.append(dirpath + '/' + names)
    return files

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_model(device, cl):
    
    unet = smp.Unet('resnet152', classes=cl, activation=None, encoder_weights='imagenet')

    if t.cuda.is_available():
        unet.cuda()         
    
    unet = unet.to(device)
    return unet


def import_data_jem(args, batch_sz):
    
    root = pathlib.Path('./')
    '''
    inputs = get_files('../../input_data/raw_john_handy/')
    inputs_full = get_files('../../input_data/raw_john_cam/')
    targets = get_files('../../input_data/mask_john_handy/')
    targets_full = get_files('../../input_data/mask_john_cam/')
    samples = get_files( '../../input_data/raw_john_handy/')
    '''
    inputs = get_files('./input_data/raw_john_handy/')
    inputs_full = get_files('./input_data/raw_john_cam/')
    targets = get_files('./input_data/mask_john_handy/')
    targets_full = get_files('./input_data/mask_john_cam/')
    samples = get_files( './input_data/raw_john_handy/')
    
    split = 0.8  

    transforms_train = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCropTrain_JEM()
        ])
    transforms_valid = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCropVal_JEM()
        ])
    transforms_sample = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01()
        ])

    # train dataset
    dataset_train = SegmentationDataSet(inputs=inputs,
                                        targets=targets,
                                        transform=transforms_train)


    # validation dataset
    dataset_valid = SegmentationDataSet(inputs=inputs_full,
                                        targets=targets_full,
                                        transform=transforms_valid)

    #  sampling dataset
    dataset_sample = SegmentationDataSet(inputs=inputs,
                                        targets=targets,
                                        transform=transforms_train)

    batchsize = batch_sz


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True)

    dataloader_sample = DataLoader(dataset=dataset_sample,
                                    batch_size=batchsize,
                                    shuffle=True)
    
    return dataloader_training, dataloader_validation, dataloader_sample

def import_data_ood(args, batch_sz):
    
    root = pathlib.Path('./')
    '''
    inputs = get_files('../../input_data/raw_john_handy/')
    inputs_full = get_files('../../input_data/raw_john_cam/')
    targets = get_files('../../input_data/mask_john_handy/')
    targets_full = get_files('../../input_data/mask_john_cam/')
    samples = get_files( '../../input_data/raw_john_handy/')
    '''
    inputs = get_files('./input_data/raw_john_handy/')
    inputs_full = get_files('./input_data/raw_john_cam/')
    targets = get_files('./input_data/mask_john_handy/')
    targets_full = get_files('./input_data/mask_john_cam/')
    samples1 = get_files( './input_data/raw_usa/')
    samples2 = get_files( './input_data/raw_grk_handy/')
    samples3 = get_files( './input_data/raw_grk_cam/')
    
    split = 0.8  

    transforms_train = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCropTrain_JEM()
        ])
    transforms_valid = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCropVal_JEM()
        ])
    transforms_sample = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        Resize_Sample()
        ])

    # train dataset
    dataset_train = SegmentationDataSet(inputs=inputs,
                                        targets=targets,
                                        transform=transforms_train)


    # validation dataset
    dataset_valid = SegmentationDataSet(inputs=inputs_full,
                                        targets=targets_full,
                                        transform=transforms_valid)

    #  sampling dataset
    dataset_sample1 = SegmentationDataSet(inputs=samples1,
                                        targets=targets,
                                        transform=transforms_sample)

    dataset_sample2 = SegmentationDataSet(inputs=samples2,
                                        targets=targets,
                                        transform=transforms_sample)

    dataset_sample3 = SegmentationDataSet(inputs=samples3,
                                        targets=targets,
                                        transform=transforms_sample)

    batchsize = batch_sz


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True)

    dataloader_sample1 = DataLoader(dataset=dataset_sample1,
                                    batch_size=batchsize,
                                    shuffle=True)

    dataloader_sample2 = DataLoader(dataset=dataset_sample2,
                                    batch_size=batchsize,
                                    shuffle=True)
    
    dataloader_sample3 = DataLoader(dataset=dataset_sample3,
                                    batch_size=batchsize,
                                    shuffle=True)
    
    return dataloader_training, dataloader_validation, dataloader_sample1, dataloader_sample2, dataloader_sample3

def import_data(args, batch_sz, set = 'usa'):

    root = pathlib.Path('./')
    if set == 'usa':
        inputs = get_files('./input_data/raw_usa/')
        targets = get_files('./input_data/mask_usa/')

    if set == 'john_handy':
        inputs = get_files('./input_data/raw_john_handy/')
        targets = get_files('./input_data/mask_john_handy/')

    if set == 'john_cam':
        inputs = get_files('./input_data/raw_john_cam/')
        targets = get_files('./input_data/mask_john_cam/')

    split = 0.8  

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=42,
        train_size=split,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=42,
        train_size=split,
        shuffle=True)


    if set == 'usa':
        transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCrop(),
        RandomFlip()
        ])
    else:
        transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCropVal_JEM(),
        RandomFlip()
        ])

    # train dataset
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)


    # validation dataset
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)


    batchsize = batch_sz


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True)

    
    
    return dataloader_training, dataloader_validation



def eval_classification(f, dload, device):
    corrects, losses = [], []
    for input, target in dload:
        input, target = input.to(device), target.to(device)
        logits = f(input)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, target).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == target).float().cpu().numpy()
        logits_max = logits.max(1)[1].float().cpu().numpy()
        label = target.float().cpu().numpy()
        '''
        fig, axs = plt.subplots(2)
        axs[0].imshow(logits_max[0,:,:])
        axs[1].imshow(label[0,:,:])
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, tag, args, device, dload_train, dload_valid):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "train": dload_train,
        "valid": dload_valid
        #"sample": dload_sample,
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def logits2rgb(img):
    red = [200, 0, 10]
    green = [187,207, 74]
    blue = [0,108,132]
    yellow = [255,204,184]
    black = [0,0,0]
    white = [226,232,228]
    cyan = [174,214,220]
    orange = [232,167,53]

    colours = [red, green, blue, yellow, black, white, cyan, orange]

    
    
    shape = np.shape(img)
    h = int(shape[0])
    w = int(shape[1])
    col = np.zeros((h, w, 3))
    unique = np.unique(img)
    for i, val in enumerate(unique):
        mask = np.where(img == val)
        for j, row in enumerate(mask[0]):
            x = mask[0][j]
            y = mask[1][j]
            col[x, y, :] = colours[int(val)]

    return col.astype(int)
'''
def iou(outputs: t.Tensor, labels: t.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = 1e-6
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs.astype(np.int32) & labels.astype(np.int32))#.sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs.astype(np.int32) | labels.astype(np.int32)).sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou
'''

def mIOU(pred, label, num_classes=8):
    
    iou_list = list()
    present_iou_list = list()

    #pred = pred.view(-1)
    #label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            #inters =np.logical_and(pred_inds, target_inds).sum().item()
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        miou = np.mean(present_iou_list)
    return miou