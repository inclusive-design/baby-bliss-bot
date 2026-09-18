from PIL import Image
import os
import sys
from common_funcs import get_max_dimensions

"""
This script synchronizes the size of all PNG and JPG files in the input directory.
It first finds the maximum dimension (either width or height) among all the input images.
Then it loops through the image directory to perform these operations for every image:
1. Transform the image to grayscale and find the background color of this image using
the color code at the pixel (1, 1);
2. Create a square canvas with the maximum dimension as its width and height. The color of
the canvas is the background color observed at the previous step;
3. Copy each input image onto the center of the canvas, without changing the size of the
input image. This ensures that each output image has the same maximum dimension and is
centered in the canvas.
Finally, all output images are saved in the specified output directory.

Usage: python image_size_sync.py [input_dir] [output_dir]
Parameters:
  input_dir: The directory where the original images are located.
  output_dir: The directory where the output images will be saved.
Return: None

Example: python image_size_sync.py images/ output/

"""


def fit_image_to_canvas(img_path, canvas_size, output_img_path):
    """
    Fits a PNG or JPG file into a larger square canvas with transparent background.
    The center of the image is placed in the center of the canvas.
    The output image is saved in the given path.

    Parameters:
        img_path (str): The path to the input image file.
        canvas_size (int): The desired size of the square canvas.
        output_img_path (str): The path to the output image file to be saved.

    Returns:
        None.
    """

    # Open the input image
    img = Image.open(img_path)

    # Get the width and height of the input image
    width, height = img.size

    # Transform the input image to grayscale
    img_gray = img.convert('L')

    # Get the color of the input image at (1, 1). This color is used to create a canvas
    # with the same background color
    pixel_color = img_gray.getpixel((1, 1))

    # Calculate the size of the output canvas
    canvas_dimension = (canvas_size, canvas_size)

    # Create a new transparent canvas of the required size
    canvas = Image.new('L', canvas_dimension, pixel_color)

    # Calculate the position to paste the input image onto the canvas
    x = int((canvas_size - width) / 2)
    y = int((canvas_size - height) / 2)

    # Paste the input image onto the canvas
    canvas.paste(img, (x, y))

    canvas.save(output_img_path)


# Check if the correct number of arguments were provided
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} image_directory output_directory")
    print(f"Example: python {sys.argv[0]} images/ output/")
    sys.exit(1)

# Parse the input arguments
image_dir = sys.argv[1]
output_dir = sys.argv[2]

# Check if the input directory exists
if not os.path.isdir(image_dir):
    print(f"Error: Input directory {image_dir} not found")
    sys.exit(1)

# Create the output directory if it doesn't exist
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

dimensions = get_max_dimensions(image_dir)

size_to_fit = dimensions[0] if dimensions[0] > dimensions[1] else dimensions[1]
print("The size to synchronize for every image: ", size_to_fit)

# Iterate over all files in the input directory
for filename in os.listdir(image_dir):
    # Check if the file is an image
    if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):
        continue

    # Get the original file name and extension
    file_name, file_ext = os.path.splitext(os.path.basename(filename))
    input_file = os.path.join(image_dir, filename)

    # Save the output image with the appropriate file name and extension
    output_filename = f"{file_name}-{size_to_fit}x{size_to_fit}{file_ext}"

    fit_image_to_canvas(input_file, size_to_fit, os.path.join(output_dir, output_filename))
    print("Processed ", input_file)
