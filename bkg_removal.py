import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

# sharpness coordinates are (column, row)

def calculate_local_sharpness(image_patch):
    # Convert the patch to grayscale if it's not already
    if len(image_patch.shape) == 3:
        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian sharpness
    laplacian_result = cv2.Laplacian(image_patch, cv2.CV_64F)
    
    # Check if the Laplacian result is a valid array
    if laplacian_result is None or laplacian_result.size == 0:
        return 0

    # Calculate local sharpness as the variance of the Laplacian result
    sharpness = laplacian_result.var()
    
    return sharpness

def find_sharpest_region(image_path, grid_size):
    try:
        image = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if image is None:
            raise Exception("Error: Unable to load the image.")

        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the Laplacian sharpness
        laplacian_result = cv2.Laplacian(image_gray, cv2.CV_64F)
        
        # Check if the Laplacian result is a valid array
        if laplacian_result is None or laplacian_result.size == 0:
            raise Exception("Error: Invalid Laplacian result.")

        # Find the coordinates of the maximum Laplacian sharpness (global maximum)
        global_max_coord = np.unravel_index(np.argmax(np.abs(laplacian_result)), laplacian_result.shape)

        # Ensure that global_max_coord is a tuple with valid coordinates
        if not global_max_coord or len(global_max_coord) != 2:
            print(f"Error: Invalid coordinates found for {image_path}")
            global_max_coord = (0, 0)  # Return a default value

        # Calculate overall sharpness as the variance of the Laplacian result (global sharpness)
        overall_sharpness = laplacian_result.var()

        # Calculate local sharpness in the region around the global maximum
        i, j = global_max_coord
        local_patch = image_gray[i-grid_size//2:i+grid_size//2, j-grid_size//2:j+grid_size//2]
        local_sharpness = calculate_local_sharpness(local_patch)
        
        image_with_target = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        return (j, i), overall_sharpness, local_sharpness, image_with_target
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return (0, 0), 0, 0, None  # Return a default value

def process_images_in_folder(folder_path=".", target_size=100, grid_size=10):
    valid_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    
    visual_regions_folder = os.path.join(folder_path, "visual_regions")
    
    # # Delete the visual_regions folder if it already exists
    # if os.path.exists(visual_regions_folder):
    #     for file_name in os.listdir(visual_regions_folder):
    #         file_path = os.path.join(visual_regions_folder, file_name)
    #         os.remove(file_path)
    #     os.rmdir(visual_regions_folder)
    
    # os.makedirs(visual_regions_folder)
    
    output_file_path = os.path.join(folder_path, "sharpness_and_coordinates.txt")
    
    with open(output_file_path, "w") as output_file:
        output_file.write("Filename\tSharpest_Coord\tOverall_Sharp\tLocal_Sharp\n")
    
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, filename)
                max_coord, overall_sharpness, local_sharpness, image_with_target = find_sharpest_region(image_path, grid_size)
                
                # Skip processing if an error occurred in find_sharpest_region
                if image_with_target is None:
                    continue

                # Standardize tones of the image before drawing the target sign
                image_with_target = cv2.normalize(image_with_target, None, 0, 255, cv2.NORM_MINMAX)

                # Draw grid and calculate sharpness in each square
                sharpness_map = np.zeros_like(image_with_target, dtype=float)

                for i in range(0, image_with_target.shape[0] - grid_size, grid_size):
                    for j in range(0, image_with_target.shape[1] - grid_size, grid_size):
                        patch = image_with_target[i:i+grid_size, j:j+grid_size]
                        sharpness = calculate_local_sharpness(patch)
                        sharpness_map[i:i+grid_size, j:j+grid_size] = sharpness

                # Draw rectangles on the photo with colors corresponding to sharpness
                image_with_grid = cv2.cvtColor(image_with_target, cv2.COLOR_BGR2GRAY)
                image_with_grid = cv2.cvtColor(image_with_grid, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing color rectangles
                for i in range(0, image_with_target.shape[0] - grid_size, grid_size):
                    for j in range(0, image_with_target.shape[1] - grid_size, grid_size):
                        sharpness = int(np.mean(sharpness_map[i:i+grid_size, j:j+grid_size]))
                        color = (255 - sharpness, 255 - sharpness, 255 - sharpness)
                        cv2.rectangle(image_with_grid, (j, i), (j+grid_size, i+grid_size), color, -1)

                # Save the image with the grid and target to the visual_regions folder
                output_visual_path = os.path.join(visual_regions_folder, filename.replace(".JPG", "_mask2.JPG"))
                save_best = 0.5
                threshold = 256*save_best
                image_with_grid[image_with_grid < threshold] = 0
                image_with_grid[image_with_grid >= threshold] = 255

                label_img = label(image_with_grid)
                regions = regionprops(label_img)
                regions = sorted(regions, key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    for rg in regions[2:]:
                        image_with_grid[rg.coords[:,0], rg.coords[:,1]] = 0
                image_with_grid[image_with_grid!=0] = 255

                # plt.figure()
                # plt.imshow(image_with_grid)

                # plt.show()
                cv2.imwrite(output_visual_path, image_with_grid)
                
                # Write the coordinates and sharpness to the output file
                output_file.write(f"{filename}\t{max_coord}\t{overall_sharpness}\t{local_sharpness}\n")
                
                print(f"For {filename}, the sharpest region is at coordinates: {max_coord}. Visualized image saved to: {output_visual_path}")

if __name__ == "__main__":
    folder = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\gsam\\Results\\foreground_white"
    process_images_in_folder(folder)
