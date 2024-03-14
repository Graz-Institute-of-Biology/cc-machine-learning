#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import segmentation_models_pytorch as smp
import requests
from io import BytesIO
import inspect
import traceback
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
import json

from config import CONFIG
import numpy as np

# print(CONFIG)

def update_analysis(analysis_id, token=None, status=None, completed=True, error=""):
    """update analysis entry in django backend
    """
    # analyses_url = 'http://django:8000/api/v1/analyses/{0}/'.format(str(analysis_id)) #localhost:8000 or django:8000 (if using docker)
    analyses_url = '{0}/api/v1/analyses/{1}/'.format(CONFIG["HOST"], str(analysis_id)) #localhost:8000 or django:8000 (if using docker)

    payload = {
        "completed" : completed,
        "errors" : error,
        "end_time" : datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if status:
        payload["status"] = status

    print("sending analysis-PATCH-request...")
    header = {"Authorization":"Token {0}".format(token)}
    response = requests.patch(analyses_url, data=payload, headers=header)
    print("done")

def get_image(img_id):
    """
    Get image from django backend
    """
    # image_url = 'http://django:8000/api/v1/images/{0}/'.format(img_id) #localhost:8000 or django:8000 (if using docker)
    image_url = '{0}/api/v1/images/{1}/'.format(CONFIG["HOST"], img_id) #localhost:8000 or django:8000 (if using docker)

    response = requests.get(image_url)
    encoded = response.content
    decoded = json.loads(encoded.decode())
    return decoded

def get_class_distributions(img, ontology, ignore_zero=True, debug=False):
    """
    Get class distributions from image
    if ignore_zero is True, ignore class 0 (Background) and calculate distribution
    in relation to the sum of all other classes
    """
    img = np.array(img)

    # if ignore_zero:
    #     # denom = np.sum(img[img != 0])
    #     print("Ignoring class 0 (Background))")
    #     print("sum: ", denom)
    # else:
    #     denom = np.sum(img)

    class_distributions = ontology.copy()
    class_names = list(ontology.keys())
    
    # set all class distributions to 0
    if not debug:
        class_distributions = dict(zip(class_distributions, np.zeros(len(class_distributions))))

    # for debugging set random class distributions
    elif debug:
        random_vals = np.random.random(len(class_distributions))
        random_vals = random_vals/np.sum(random_vals)
        class_distributions = dict(zip(class_distributions, random_vals))
        # for key in enumerate(class_distributions):
        #     class_distributions[key] = np.random.rand()

    unique, counts = np.unique(img, return_counts=True)
    denom = np.sum(counts[1:])
    print("Unique: ", unique)
    print("Counts: ", counts)
    for i in range(len(unique)):
        class_code = unique[i]
        name = class_names[class_code]
        class_distributions[name] = round(counts[i]/denom, 5)

    del class_distributions['background']
    class_distributions = str(class_distributions)
    print("Class Dist.:")
    print(class_distributions)
    return class_distributions

def send_result(color_coded_img, categorically_coded_img, ontology, item):
    """
    Send result to to django backend
    """

    source_model_url = item.ml_model_path
    parent_file_url = item.file_path
    parent_img_id = item.parent_img_id
    ml_model_id = item.ml_model_id
    debug = item.debug
    class_distributions = get_class_distributions(categorically_coded_img, ontology, debug=debug)

    # image_data = get_image(parent_img_id)
    dataset = item.dataset_id
    
    img_byte_arr = io.BytesIO()
    color_coded_img.save(img_byte_arr, format='png')

    payload = {
        "name" : 'test',
        "owner" : 'admin',
        "description" : "test description",
        "slug" : 'test',
        "dataset" : dataset,
        "parent_image" : parent_img_id,
        "source_model" : ml_model_id,
        "source_model_url" : source_model_url,
        "parent_image_url" : parent_file_url,
        "class_distributions" : class_distributions,
    }
    file = {'mask': ('image.jpg', img_byte_arr.getvalue(), 'image/png')}
    
    print("sending mask-POST-request...")
    mask_api_url = '{0}/api/v1/masks/'.format(CONFIG["HOST"]) #localhost:8000 or django:8000 (if using docker)
    header = {"Authorization":"Token {0}".format(item.token)}
    response = requests.post(mask_api_url, data=payload, files=file, headers=header)
    print("done")

    pass

def plot_save_mask(mask, ontology):

    print("plot & save results...")
    mask = np.array(mask)
    labels = list(ontology.keys())
    colors_hex = list(ontology.values())

    # colors_hex = ["#000000","#1cffbb", "#00bcff","#0059ff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
    col = ListedColormap(colors_hex)
    bounds = np.arange(len(colors_hex))
    norm = BoundaryNorm(bounds, col.N)  
    # img_arr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    fig = plt.figure(figsize=mask.shape[:2][::-1], dpi=1)

    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    
    plt.imshow(mask, cmap=col, interpolation='nearest', norm=norm)
    plt.axis('off')
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    pil_img = Image.fromarray(img_arr)

    print("done")
    
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

def predict(item, debug) -> np.ndarray:
    """
    """
    model_url = item.ml_model_path
    image_url = item.file_path
    add_parent_dir()
    img = load_image(image_url)
    trainer, ontology = create_trainer_object(model_url)
    mask_pred = trainer.predict_whole_image(img, debug=debug)
    # mask_pred = img
    print("Img: {0} finished".format(image_url))
    return mask_pred, ontology


def manage_prediction_request(item):

    try:
        debug = item.debug
        update_analysis(item.analysis_id, item.token, completed=False, status="processing")
        mask_pred, ontology = predict(item=item, debug=debug)
        color_coded_mask = plot_save_mask(mask_pred, ontology)
        update_analysis(item.analysis_id, item.token, completed=True, status="processed/sending result")
        send_result(color_coded_mask, mask_pred, ontology, item)
        update_analysis(item.analysis_id, item.token, completed=True, status="processed & saved")
        return {
            "error": False,
        }
    except Exception as e:
            print("ERROR:")
            error = traceback.format_exc()
            print(error)
            update_analysis(item.analysis_id, item.token, completed=False, error=error)
            return {
            "error": True,
        }