'''
This script filters out all Bliss single characters from a directory with all Bliss symbols.

Steps:
1. Read a tab separated .tsv file exported from [BCI-AV Easter Characters](https://docs.google.com/spreadsheets/d/1t1x1UFuJC1hpjrxdXKi19Tk_Tv-9GVQWSA4sN2FScv4/edit#gid=138588066).
that has all the information of Bliss single characters.
2. Pad the values of the first column to a length of 5, then add ".png" extension to these padded values
3. find these png files in a directory that has all bliss symbols in PNG format
4. The found png files are copied to the target directory. If a png file is not found, report an error.

Usage: python script_name.py tsv_file_path all_bliss_symbol_dir target_dir
Parameters:
  tsv_file_path: The path to the .tsv file to be read.
  all_bliss_symbol_dir: The path to the directory where all Bliss symbol images are located.
  target_dir: The path to the directory where matched symbol images will be copied to.
Return: All Bliss single characters are copied into the target directory.

Example: python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars
'''

import os
import shutil
import sys

# Get the file paths from the command line arguments
tsv_file_path = sys.argv[1]
all_bliss_symbol_dir = sys.argv[2]
target_dir = sys.argv[3]

# Create the output directory if it doesn't exist
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

# Open the .tsv file for reading
with open(tsv_file_path, 'r') as tsv_file:
    # Skip the first row (header)
    next(tsv_file)

    # Iterate over the rows in the .tsv file
    for row in tsv_file:
        # Split the row into columns
        columns = row.strip().split('\t')

        # Get the value in the first column
        value = columns[0]

        # Pad the value with "0" to a length of 5
        padded_value = value.zfill(5)

        # Add the ".png" extension to the padded value
        png_filename = padded_value + ".png"

        # Check if the png file exists in the png directory
        png_file_path = os.path.join(all_bliss_symbol_dir, png_filename)
        if os.path.exists(png_file_path):
            # Copy the png file to the matched png directory
            matched_png_file_path = os.path.join(target_dir, png_filename)
            shutil.copy(png_file_path, matched_png_file_path)
        else:
            # Report an error if the png file is not found
            print("Error: {} not found in {}".format(png_filename, all_bliss_symbol_dir))
