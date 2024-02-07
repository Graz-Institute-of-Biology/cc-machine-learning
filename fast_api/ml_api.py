#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import segmentation_models_pytorch as smp
import requests
from io import BytesIO
import inspect
import os
import sys
import cv2
import io
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# from config import CONFIG
import numpy as np

def update_analysis(id,completed=True):
    """update analysis entry in django backend
    """
    analyses_url = 'http://django:8000/api/v1/analyses/' + str(id) + '/'

    payload = {
        "completed" : completed,
        "end_time" : datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    print("Update analysis...")
    response = requests.patch(analyses_url, data=payload)
    print(response.text)
    print("done")

def send_result(img, source_model_url, parent_file_url):
    """
    Send result to to django backend
    """
    print("sending request...")
    mask_api_url = 'http://django:8000/api/v1/masks/'
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='png')

    print(type(img_byte_arr))

    payload = {
        "name" : 'test',
        "owner" : 'admin',
        "description" : "test description",
        "slug" : 'test',
        "dataset" : 2,
        "parent_image" : 1,
        "source_model" : 7,
        "source_model_url" : source_model_url,
        "parent_image_url" : parent_file_url
    }
    file = {'mask': ('image.jpg', img_byte_arr.getvalue(), 'image/png')}    # print(image)
    
    response = requests.post(mask_api_url, data=payload, files=file)
    print(response.text)
    print("done")

    pass

def plot_save_mask(mask, ontology):

    print("plotting results...")
    mask = np.array(mask)
    labels = list(ontology.keys())
    colors_hex = list(ontology.values())

    # colors_hex = ["#000000","#1cffbb", "#00bcff","#0059ff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
    col = ListedColormap(colors_hex)
    bounds = np.arange(len(colors_hex))
    norm = BoundaryNorm(bounds, col.N)  
    print(mask.shape)
    # img_arr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    fig = plt.figure(figsize=mask.shape[:2][::-1], dpi=1)

    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    
    plt.imshow(mask, cmap=col, interpolation='nearest', norm=norm)
    plt.axis('off')
    
    print("saving to buffer...")
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    print("done")
    pil_img = Image.fromarray(img_arr)
    
    return pil_img

def prepare_load_model(model_url):
    encoder = 'mit_b5'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'
    class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes=len(class_values), 
        activation=activation,
    )

    response = requests.get(model_url)
    # print(response.content)
    checkpoint = torch.load(BytesIO(response.content), map_location=torch.device('cpu'))
    ontology = checkpoint['ontology']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, ontology

def add_parent_dir():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    if parentdir not in sys.path:
        sys.path.insert(0, parentdir)

def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def create_trainer_object(model_url):
    from train_smp import Trainer
    encoder = 'mit_b5'
    print("Creating trainer object...")
    trainer = Trainer(encoder=encoder, load_config=False, device='cpu')
    print("Done")
    print("Loading model...")
    model, ontology = prepare_load_model(model_url)
    trainer.model = model
    print("Done")
    return trainer, ontology

def predict(package: dict, input: list) -> np.ndarray:
    """
    """
    print(package)
    model_url = package['model_path']
    image_url = package['file_path']
    add_parent_dir()
    print("loading image...")
    img = load_image(image_url)
    print("Image loaded")
    print("Creating trainer object...")
    trainer, ontology = create_trainer_object(model_url)
    mask_pred = trainer.predict_whole_image(img)
    # mask_pred = img
    color_coded_mask = plot_save_mask(mask_pred, ontology)
    print("Done")
    print("predicting...")
    print("FINISHED")
    return mask_pred, color_coded_mask
