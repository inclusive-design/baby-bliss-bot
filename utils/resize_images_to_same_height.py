from PIL import Image
import os
import sys

'''
This script resizes all images in a directory to the same height. The resized images are saved into a target directory.

Usage: python resize_images_to_same_height.py <image_dir> <target_height> <target_dir>
Parameters:
  image_dir: The directory with all images
  target_height: The target height to resize all images to
  target_dir: The target directory to save resized images
Return: None

Example: python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216
'''


def resize_images(source_dir, target_height, target_dir):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through all files in source directory
    for filename in os.listdir(source_dir):
        # Check if file is an image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):

            # Open image file
            with Image.open(os.path.join(source_dir, filename)) as img:
                # Get current width and height
                width, height = img.size

                # Calculate new width based on target height
                ratio = target_height / height
                new_width = int(width * ratio)

                # Resize image
                resized_img = img.resize((new_width, target_height))

                # Save resized image to target directory
                resized_img.save(os.path.join(target_dir, filename))
                print(f"{filename} has been resized and saved to {target_dir}")


if __name__ == "__main__":
    image_dir = sys.argv[1]
    target_height = int(sys.argv[2])
    target_dir = sys.argv[3]

    resize_images(image_dir, target_height, target_dir)
