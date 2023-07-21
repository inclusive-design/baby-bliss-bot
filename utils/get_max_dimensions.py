import sys
from common_funcs import get_max_dimensions

"""
This script finds the maximum width and maximum height of all images in a folder,
along with a list of image filenames that have the maximum width and maximum height.
It also finds the second maximum width and second maximum height, along with their respective
lists of image filenames.

Usage: python get_max_dimensions.py [image_directory]
Parameter:
  image_directory: The path to the directory containing the images.
Return: tuple: A tuple containing:
  * the maximum width (int)
  * maximum height (int)
  * a list of filenames of images with maximum width (list)
  * a list of filenames of images with maximum height (list)
  * the second maximum width (int), the second maximum height (int)
  * a list of filenames of images with the second maximum width (list)
  * a list of filenames of images with the second maximum height (list)

Example: python get_max_dimensions.py images/
"""

# Check if the correct number of arguments were provided
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} image_directory")
    print(f"Example: python {sys.argv[0]} images/")
    sys.exit(1)

# Parse the input arguments
image_dir = sys.argv[1]

results = get_max_dimensions(image_dir)
print("The max width is: ", results[0])
print("The list of images with the max width is: ", results[2])
print("The max height is: ", results[1])
print("The list of images with the max height is: ", results[3])
print("The second max width is: ", results[4])
print("The list of images with the second max width is: ", results[6])
print("The second max height is: ", results[5])
print("The list of images with the second max height is: ", results[7])
