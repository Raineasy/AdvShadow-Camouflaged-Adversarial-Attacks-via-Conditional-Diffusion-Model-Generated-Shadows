import os
import random
import shutil


def pick_images_and_masks(source_dir, mask_dir, output_dir, num_images=500):
    """
    Randomly pick images from source_dir and their corresponding masks from mask_dir.

    Args:
        source_dir (str): Directory containing the original images.
        mask_dir (str): Directory containing the mask images.
        output_dir (str): Directory to store selected images and masks.
        num_images (int): Number of images to select.
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Randomly select images
    selected_images = random.sample(all_files, num_images)

    # Iterate over selected images and find corresponding masks
    for image_name in selected_images:
        mask_name = f"mask_{image_name}"
        print(mask_name)
        image_path = os.path.join(source_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        # Check if the mask exists
        if os.path.exists(mask_path):
            # Copy image and mask to the output directory
            shutil.copy(image_path, os.path.join(output_dir, image_name))
            shutil.copy(mask_path, os.path.join(output_dir, mask_name))
        else:
            print(f"Mask not found for image {image_name}")


# Example usage
source_directory = 'E:\\tree\\ddim\\images'
mask_directory = 'E:/tree/ddim/images_mask'
output_directory = 'E:/tree/ddim/figure'

pick_images_and_masks(source_directory, mask_directory, output_directory)
