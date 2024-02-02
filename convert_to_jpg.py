from PIL import Image
import os

def convert_to_jpg(png_file, jpg_file):
    image = Image.open(png_file).convert('RGB')
    image.save(jpg_file, 'JPEG')

def convert_folder_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            png_file = os.path.join(folder_path, filename)
            jpg_file = os.path.join(folder_path, filename.replace(".png", ".jpg"))
            convert_to_jpg(png_file, jpg_file)


if __name__ == "__main__": 
    folder_path = 'test\\g_street_view'
    convert_folder_to_jpg(folder_path)
