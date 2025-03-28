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
from PIL import Image, ImageColor
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
from pathlib import Path

from config import CONFIG
import numpy as np

# print(CONFIG)

def update_analysis(analysis_id, token=None, status=None, completed=True, error=""):
    """update analysis entry in django backend
    """
    # analyses_url = 'http://django:8000/api/v1/analyses/{0}/'.format(str(analysis_id)) #localhost:8000 or django:8000 (if using docker)
    analyses_url = '{0}/django/api/v1/analyses/{1}/'.format(CONFIG["HOST"], str(analysis_id)) #localhost:8000 or django:8000 (if using docker)

    print("URL: ", analyses_url)
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
    print(response)
    print("done")

def get_image(img_id):
    """
    Get image from django backend
    """
    # image_url = 'http://django:8000/api/v1/images/{0}/'.format(img_id) #localhost:8000 or django:8000 (if using docker)
    image_url = '{0}/django/api/v1/images/{1}/'.format(CONFIG["HOST"], img_id) #localhost:8000 or django:8000 (if using docker)

    response = requests.get(image_url)
    encoded = response.content
    decoded = json.loads(encoded.decode())
    return decoded

def check_color(color):
    if color[0] == "#":
        return ImageColor.getrgb(color)
    else:
        return color

def get_class_color_dict(ontology):

    class_color_dict = {}
    for i, key in enumerate(ontology.keys()):
        # if key == "background":
        #     print("Ignoring class 'background'...")
        #     continue
        rgb_color = check_color(ontology[key])
        class_color_dict[i] = {"name": key, "color": rgb_color}

    return class_color_dict

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

    if class_names[0] == "0":
        class_names = [class_distributions[key]["name"] for key in class_distributions]

    print(class_distributions)
    
    # set all class distributions to 0
    if not debug:
        class_distributions = dict(zip(class_names, np.zeros(len(class_names))))

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

    # del class_distributions['background']
    # class_distributions = str(class_distributions)
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
    if not list(ontology.keys())[0] == "0":
        class_color_dict = get_class_color_dict(ontology)
    else:
        class_color_dict = ontology
    class_distributions = json.dumps({"class_distributions": class_distributions, "class_colors": class_color_dict})

    # image_data = get_image(parent_img_id)
    dataset = item.dataset_id
    
    img_byte_arr = io.BytesIO()
    color_coded_img.save(img_byte_arr, format='png')

    name = Path(parent_file_url).stem
    slug = name.lower()

    if len(slug) > 50:
        slug = slug[:50]

    payload = {
        "name" : name,
        "owner" : 'admin',
        "description" : "empty",
        "slug" : slug,
        "dataset" : dataset,
        "parent_image" : parent_img_id,
        "source_manual" : False,
        "source_model" : ml_model_id,
        "source_model_url" : source_model_url,
        "parent_image_url" : parent_file_url,
        "class_distributions" : class_distributions,
    }

    mask_file_name = name + "_mask.png"
    file = {'mask': (mask_file_name, img_byte_arr.getvalue(), 'image/png')}
    
    print("sending mask-POST-request...")
    mask_api_url = '{0}/django/api/v1/masks/'.format(CONFIG["HOST"]) #localhost:8000 or django:8000 (if using docker)
    header = {"Authorization":"Token {0}".format(item.token)}
    response = requests.post(mask_api_url, data=payload, files=file, headers=header)
    print("Done")

    pass

def get_hex_from_dict(ontology):
    class_values = ontology.keys()
    colors_hex = []
    for key in class_values:
        color_rgb = ontology[key]["color"]
        color_rgb = np.array(color_rgb)/255
        color_hex = matplotlib.colors.rgb2hex(color_rgb)
        colors_hex.append(color_hex)
    
    return colors_hex

def plot_save_mask(mask, ontology):
    """ saves color mask


    Args:
        mask (np array)):   2 dim array with predictions [[0011][23NN][...]] with N classes
        ontology (dict):    dictionary with either class values { "0" : "HEX1" , "1" : "HEX2" , ... , "N" : HEXN"] 
                            or class names {"background" : "HEX1" "class1" : "HEX2", ... , "classN" : "HEXN"} 
                            as keys and color values as values

    Returns:
        _type_: _description_
    """

    print("plot & save results...")
    mask = np.array(mask)
    labels = list(ontology.keys())
    if labels[0] == "0":
        print("GET HEX FROM DICT")
        colors_hex = get_hex_from_dict(ontology)
    else:
        print("list(ontology.values())")
        colors_hex = list(ontology.values())

    print("COLORS: ", colors_hex)
    # colors_hex = ["#000000","#1cffbb", "#00bcff","#0059ff", "#2601d8", "#ff00c3", "#FF4A46", "#ff7500", "#928e00"]
    col = ListedColormap(colors_hex)
    bounds = np.arange(len(colors_hex) +1 ) - 0.5
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

def prepare_load_model(model_url, num_classes, encoder='mit_b5'):
    encoder_weights = 'imagenet'
    activation = 'sigmoid'

    if 'localhost' in model_url:
        model_url = model_url.replace('localhost', 'django')

    response = requests.get(model_url)
    # print(response.content)
    checkpoint = torch.load(BytesIO(response.content), map_location=torch.device('cpu'))
    ontology = checkpoint['ontology']
    print("Using ontology: ", ontology)
    crop_size = None
    try:
        crop_size = checkpoint['grid_size']
        num_classes = checkpoint['num_classes']
        activation = checkpoint['activation']
        encoder_weights = checkpoint['encoder_weights']
        encoder = checkpoint['encoder']
        print("Using model settings from checkpoint: \n")
        print("Crop size: ", crop_size)
        print("Num classes: ", num_classes)
        print("Activation: ", activation)
        print("Encoder weights: ", encoder_weights)
        print("Encoder: ", encoder)
    except KeyError:
        print("No model settings found in checkpoint, trying default values")


    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes=num_classes, 
        activation=activation,
        )

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, ontology, crop_size

def add_parent_dir():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    if parentdir not in sys.path:
        sys.path.insert(0, parentdir)

def load_image(image_url):
    if 'localhost' in image_url:
        image_url = image_url.replace('localhost', 'django')
        print("replaced:", image_url)
    else:
        print("not replaced:", image_url)
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def create_trainer_object(model_url, num_classes):
    from train_smp import Trainer
    encoder_list = ['mit_b5', 'mit_b3', 'mit_b1']
    print("Creating trainer object...")
    trainer = None
    for encoder in encoder_list:
        try:
            trainer = Trainer(encoder=encoder, load_config=False, device='cpu')
            model, ontology, crop_size = prepare_load_model(model_url, num_classes, encoder=encoder)
            break
        except Exception as e:
            print("Error: ", e)
            print("Trying next encoder...")

    if not trainer:
        raise Exception("No encoder found for model: {0}".format(model_url))
    print("Done")
    print("Loading model...")
    trainer.model = model
    trainer.class_values = list(np.arange(num_classes))
    if crop_size:
        trainer.size = crop_size
    print("Done")
    return trainer, ontology

def predict(item, debug) -> np.ndarray:
    """
    """
    model_url = item.ml_model_path
    image_url = item.file_path
    num_classes = item.num_classes
    add_parent_dir()
    img = load_image(image_url)
    trainer, ontology = create_trainer_object(model_url, num_classes)
    mask_pred = trainer.predict_whole_image(img, debug=debug)
    # mask_pred = img
    print("Img: {0} finished".format(image_url))
    return mask_pred, ontology


def manage_prediction_request(item):
    print("HOST: ", CONFIG["HOST"])
    print("ITEM: ", item)
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