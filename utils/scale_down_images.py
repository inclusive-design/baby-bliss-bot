import os
import sys
from PIL import Image

"""
Copyright (c) 2023, Inclusive Design Institute

Licensed under the BSD 3-Clause License. You may not use this file except
in compliance with this License.

You may obtain a copy of the BSD 3-Clause License at
https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE
"""

"""
This script scales down JPG and PNG images in a directory to a specified size
while maintaining their aspect ratios. The output images are saved in a new
directory. If the output directory doesn't exist, it will be created.

Usage: python scale_down_images.py [input_dir] [output_dir] [new_size]
Parameters:
  input_dir: The directory where the original images are located.
  output_dir: The directory where the output images will be saved.
  new_size: The desired size of the scaled down images, in the format "widthxheight".
Return: None

Example: python scale_down_images.py images/ scaled_down_images/ 128x128
"""

# Check if the correct number of arguments were provided
if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} input_directory output_directory size")
    print(f"Example: python {sys.argv[0]} images/ scaled_down_images/ 128x128")
    sys.exit(1)

# Parse the input arguments
input_dir = sys.argv[1]
output_dir = sys.argv[2]
size_str = sys.argv[3]

# Check if the input directory exists
if not os.path.isdir(input_dir):
    print(f"Error: Input directory {input_dir} not found")
    sys.exit(1)

# Create the output directory if it doesn't exist
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Parse the size string into width and height
try:
    width, height = map(int, size_str.split('x'))
except ValueError:
    print(f"Error: Invalid size string {size_str}. Must be in the format 'widthxheight'")
    sys.exit(1)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):
        continue

    # Open the image file
    filepath = os.path.join(input_dir, filename)
    with Image.open(filepath) as img:
        # Scale down the image while maintaining its aspect ratio
        img.thumbnail((width, height))

        # Construct the output filename and path
        output_filename = f"{os.path.splitext(filename)[0]}_{width}x{height}{os.path.splitext(filename)[1]}"
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the output image to the output directory
        img.save(output_filepath)
        print(f"Scaled down {filename} to {output_filename}")
