from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from PIL import ImageColor, Image
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

# class_dict = {	    "background" : 0,
#                         "liverwort" : 1,
#                         "moss" : 2,
#                         "cyanosliverwort" : 3,
#                         "cyanosmoss" : 4,
#                         "lichen" : 5,
#                         "barkdominated" : 6,
#                         "cyanosbark" : 7,
#                         "other" : 8,
#                     }

class_dict = {	    "Background" : 0,
                        "Bryophytes" : 2,
                        "Bryophytes & Cyanos" : 4,
                        "Lichen" : 5,
                        "Bark-Dominated" : 6,
                        "Bark & Cyanos" : 7,
                        "Other" : 8,
                    }

class_values = list(class_dict.values())

def save_images_to_pdf(image, mask, title=None):
        """
        create plots for training and validation loss,
        visualize images in a row: input image, ground truth and prediction
        """

        labels = list(class_dict.keys())

        # colors_hex = ["#000000",
        #               "#1cffbb",
        #               "#00bcff",
        #               "#0059ff",
        #               "#2601d8",
        #               "#ff00c3",
        #               "#FF4A46",
        #               "#ff7500",
        #               "#928e00"]

        colors_hex = ["#000000",
                "#1cffbb",
                "#00bcff",
                "#0059ff",
                "#2601d8",
                "#ff00c3",
                "#Ff0000",
                "#FFA500",
                "#FFFF00"]
        
        col = ListedColormap(colors_hex)
        bounds = np.arange(len(colors_hex)+1)
        norm = BoundaryNorm(bounds, col.N)  

        fig = plt.figure(figsize=(16, 10))

        if title:
            fig.suptitle(title)

        plt.xticks([])
        plt.yticks([])

        plt.imshow(image)
        plt.imshow(mask, cmap=col, interpolation='nearest', norm=norm, alpha=0.5)
        patches = [ mpatches.Patch(color=colors_hex[e], label=labels[i] ) for i,e in enumerate(class_values)]
        plt.axis('off')
        plt.legend(handles=patches, bbox_to_anchor=(1.28, 1), loc=0, borderaxespad=0. )   
            
        plt.show()



if __name__ == "__main__":
    img_path = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\PhD - Institut Biologie\\Poster\\Label_example\\180223_TF_G_E_DJI_0494_part_2.JPG"
    mask_path = "C:\\Users\\faulhamm\\OneDrive - Universität Graz\\Dokumente\\Philipp\\PhD - Institut Biologie\\Poster\\Label_example\\180223_TF_G_E_DJI_0494_part_2.png"
    image = Image.open(img_path)
    mask = np.array(Image.open(mask_path))

    # print(np.unique(mask))

    save_images_to_pdf(image, mask)

