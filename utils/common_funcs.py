from PIL import Image
import os


def get_max_dimensions(folder_path):
    """
    Returns the maximum width and maximum height of all images in a folder,
    along with a list of image filenames that have the maximum width and maximum height.
    Also returns the second maximum width and second maximum height, along with their respective
    lists of image filenames.

    Parameters:
        folder_path (str): The path to the folder containing the images.

    Return:
        tuple: A tuple containing:
        * the maximum width (int)
        * maximum height (int)
        * a list of filenames of images with maximum width (list)
        * a list of filenames of images with maximum height (list)
        * the second maximum width (int), the second maximum height (int)
        * a list of filenames of images with the second maximum width (list)
        * a list of filenames of images with the second maximum height (list)
    """

    # Initialize the maximum width, second maximum width, maximum height, and second maximum height to 0
    max_width = 0
    max_height = 0
    second_max_width = 0
    second_max_height = 0

    # Initialize lists to keep track of filenames of images with maximum width, maximum height,
    # second maximum width, and second maximum height
    max_width_files = []
    max_height_files = []
    second_max_width_files = []
    second_max_height_files = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image file (in this case, .jpg, .png, or .jpeg)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # Open the image file
            image = Image.open(os.path.join(folder_path, filename))

            # Get the width and height of the image
            width, height = image.size

            # Update the maximum and second maximum width and height if necessary
            if width > max_width:
                second_max_width = max_width
                second_max_width_files = max_width_files
                max_width = width
                max_width_files = [filename]
            elif width == max_width:
                max_width_files.append(filename)
            elif width > second_max_width:
                second_max_width = width
                second_max_width_files = [filename]
            elif width == second_max_width:
                second_max_width_files.append(filename)

            if height > max_height:
                second_max_height = max_height
                second_max_height_files = max_height_files
                max_height = height
                max_height_files = [filename]
            elif height == max_height:
                max_height_files.append(filename)
            elif height > second_max_height:
                second_max_height = height
                second_max_height_files = [filename]
            elif height == second_max_height:
                second_max_height_files.append(filename)

    # Return the maximum width, maximum height, and lists of filenames of images with maximum width and maximum height,
    # the second maximum width, second maximum height, and lists of filenames of images with the second maximum width and second maximum height
    return max_width, max_height, max_width_files, max_height_files, second_max_width, second_max_height, second_max_width_files, second_max_height_files
