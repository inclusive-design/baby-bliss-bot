# Utility Scripts

This directory contains all utility scripts that are used for preparing dataset, training models etc.

## Get Bliss single characters (get_bliss_single_chars.py)

This script filters out all Bliss single characters from a directory with all Bliss symbols.

**Usage**: python script_name.py [tsv_file_path] [all_bliss_symbol_dir] [target_dir]

* *tsv_file_path*: The path to the .tsv file to be read. This file contains single characters by BCI IDs
* *all_bliss_symbol_dir*: The path to the directory where all Bliss symbol images are located.
* *target_dir*: The path to the directory where matched symbol images will be copied to.

**Example**: python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars

**Return**: None

## Resize all images to a same height (resize_images_to_same_height.py)

This script resizes all images in a directory to the same height. The resized images are saved into a target directory.

**Usage**: python resize_images_to_same_height.py [image_dir] [target_height] [target_dir]

* *image_dir*: The directory with all images
* *target_height*: The target height to resize all images to
* *target_dir*: The target directory to save resized images

**Example**: python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216

**Return**: None

## Get max image dimensions (get_max_dimensions.py)

This script finds the maximum width and maximum height of all PNG and JPG images in a directory,
along with a list of image filenames that have the maximum width and maximum height.
It also returns the second maximum width and second maximum height, along with their respective
lists of image filenames.

**Usage**: python get_max_dimensions.py [image_directory]

* *image_directory*: The path to the directory containing the images.

**Example**: python get_max_dimensions.py images/

**Return**: tuple: A tuple containing:
* the maximum width (int)
* maximum height (int)
* a list of filenames of images with maximum width (list)
* a list of filenames of images with maximum height (list)
* the second maximum width (int), the second maximum height (int)
* a list of filenames of images with the second maximum width (list)
* a list of filenames of images with the second maximum height (list)

## Scale down images (scale_down_images.py)

This script scales down JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The output images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python scale_down_images.py [input_dir] [output_dir] [new_size]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.
* *new_size*: The desired size of the scaled down images, in the format "widthxheight".

**Example**: python scale_down_images.py images/ scaled_down_images/ 128x128

**Return**: None

## Sync up image sizes (image_size_sync.py)

This script synchronizes the size of all PNG and JPG files in the input directory.
It first finds the maximum dimension (either width or height) among all the input images.
Then it loops through the image directory to perform these operations for every image:
1. Transform the image to grayscale and find the background color of this image using the color code at the pixel
(1, 1);
2. Create a square canvas with the maximum dimension as its width and height. The color of the canvas is the background
color observed at the previous step;
3. Copy each input image onto the center of the canvas, without changing the size of the input image. This ensures that
each output image has the same maximum dimension and is centered in the canvas. 
Finally, all output images are saved in the specified output directory.

**Usage**: python image_size_sync.py [input_dir] [output_dir]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.

**Example**: python image_size_sync.py images/ output/

**Return**: None
