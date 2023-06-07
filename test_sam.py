from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
from matplotlib import pyplot as plt
import numpy as np
import datetime
import yaml

def save_masks(image, masks, out_file_name):
    plt.figure(figsize=(20,20))
    plt.title("Time Elapsed: {0}".format(elapsed))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(out_file_name)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

start_time = datetime.datetime.now()

with open("paths.yaml") as f:
    yaml_file = yaml.safe_load(f)


img_path = "190223_TF_C_S_DJI_0019.JPG"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


ckpt_vit_h = yaml_file["ckpt_vit_h"]
sam = sam_model_registry["default"](checkpoint=ckpt_vit_h)
predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("test_embedding.npy", image_embedding)

# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)

end_time = datetime.datetime.now()

elapsed = end_time - start_time


# predictor.set_image(image)
# masks, _, _ = predictor.predict("Tree")