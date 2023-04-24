"""
Copyright (c) 2023, Inclusive Design Institute

Licensed under the BSD 3-Clause License. You may not use this file except
in compliance with this License.

You may obtain a copy of the BSD 3-Clause License at
https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE
"""

import json
import re
import sys

# Regular expression for finding shapes specifiers, e.g., 'W#5#2'
SHAPE_TRIAD = re.compile('[A-Z]#[0-9]+#[0-9]+')

# End of a shape and grid specification string (two versions)
GRID_END_SEQ = '##0#4*'
Z_GRID_END_SEQ = '##0#6*'

# End of entire shape string
SHAPE_END_SEQ = '%£'

# Regular expression for grid coordinates and letters
# The structure is:
# - #0              Ignore: Single hash followed by zero.
# - #number         X coordinate: Hash followed by 1 or more digits, 0 thru 9.
# - #number         X coordinate: Hash followed by 1 or more digits, 0 thru 9.
# - #number#number  Ignore:  A pair of hash-followed-by-1-or-more-digits.
# - #               Ignore:  Zero or one hashes.
# - character       Optional letter: zero or one single character.
# Examples, split up by spaces (no letter in the second example)
# ignore    X   Y   ignore  ignore  letter
# #0        #0  #10 #35#11  #       b
# #0        #0  #8  #15#9
GRID_COORD_LETTERS = re.compile('#0#([0-9]+)#([0-9]+)#[0-9]+#[0-9]+#?(.?)')

"""
Read and parse a '.wbs' file

file_path: The path to the .wbs file to process.

Example: python parse_wbs.py ./data/wbs/some_wbs_file.wb > output.json

Returns: Prints an array of JSON objects to stdout, one for each line of the
         input '.wbs' file.
"""


def make_grid_coordinates(grid_string):
    grid_entry = {}
    # Clear out all the endings
    clear_string = (
        grid_string.replace(GRID_END_SEQ, '')
        .replace(Z_GRID_END_SEQ, '')
        .replace('*', '')  # Some strings have just the asterisk at the end
        .replace(SHAPE_END_SEQ, '')
    )
    match = GRID_COORD_LETTERS.match(clear_string)
    if match:
        grid_entry['x'] = match.group(1)
        grid_entry['y'] = match.group(2)
        grid_entry['letter'] = match.group(3)
    else:
        grid_entry['x'] = ''
        grid_entry['y'] = ''
        grid_entry['letter'] = ''
    return grid_entry


def make_shape_entries(shape_string):
    shapes = []
    # Individual shapes are separated by an ampersand.
    shape_splits = shape_string.split('&')
    for asplit in shape_splits:
        shape_entry = {}
        triad = SHAPE_TRIAD.search(asplit)
        if triad:
            shape_entry['code'] = triad.group().replace('#', '-')
            grid_string = asplit[triad.end():]
            shape_entry.update(make_grid_coordinates(grid_string))
        shapes.append(shape_entry)
    return shapes


def make_wbs_entry(wbs_parts_array):
    wbs_entry = {}
    wbs_entry['identifier'] = wbs_parts_array[0]
    wbs_entry['gloss'] = wbs_parts_array[1]
    wbs_entry['pos_colour'] = wbs_parts_array[2]
    wbs_entry['country'] = wbs_parts_array[3]
    wbs_entry['shapes'] = make_shape_entries(wbs_parts_array[4])
    return wbs_entry


def read_parse_wbs_file(wbs_file):
    wbs_json_array = []
    for line in wbs_file:
        if line.isspace():
            continue
        line_parts = line.split('$')
        line_dict = make_wbs_entry(line_parts)
        wbs_json_array.append(line_dict)
    return wbs_json_array


def main():
    # Handle command line arguments
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} file_path > output_file')
        print(f'Example: {sys.argv[0]} ./data/wbs/some_wbs_file.wb > output.json')
        sys.exit(1)

    # Handle file type, at least by its extension
    if sys.argv[1].endswith('.wbs') is False:
        print(f'Input file ({sys.argv[1]}) must be a ".wbs" file')
        sys.exit(1)

    # Convert the file
    wbs_file = open(sys.argv[1], mode='r', encoding='latin-1')
    wbs_json_array = read_parse_wbs_file(wbs_file)
    print(json.dumps(wbs_json_array, indent=2, sort_keys=False))


main()
