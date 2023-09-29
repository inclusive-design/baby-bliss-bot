'''
Create the JSON file for rendering keys on the BMW Palette.

Example: python create_keys_json.py ../data/intermediate_BMW_conversion_data/symbols_in_pos.json ../data/bmw_keys.json
'''

import json
import sys
import uuid
from slugify import slugify


def incrementPosition(should_start_new_column, current_row, current_column, start_row, max_row):
    return (
        current_row + 1 if current_row < max_row + start_row - 1 else start_row,
        current_column + 1 if should_start_new_column or current_row == max_row + start_row - 1 else current_column
    )


def write_to_json(final_json, one_symbol, type, current_row, current_column):
    label = list(one_symbol.keys())[0]
    bci_av_id = list(one_symbol.values())[0]
    key = f"{slugify(label)}-{uuid.uuid4()}"
    print("key: ", key)
    final_json["cells"][key] = {
        "type": type,
        "label": label,
        "BCI-AV-ID": bci_av_id,
        "rowStart": current_row,
        "rowSpan": 1,
        "columnStart": current_column,
        "columnSpan": 1
    }
    print(f"{label}: {current_row}, {current_column}")
    return final_json


source_json_file = sys.argv[1]
output_json_file = sys.argv[2]

# Configurable options
# 1. Keys are displayed based on their POS values. Each element in `pos_in_order` array
# means keys in this/these POS catgor(ies) start in a new column.
pos_in_order = [
    "PROPN",
    "PRON",
    "NOUN",
    "VERB",
    ["ADV", "CCONJ", "INTJ", "ADJ", "SCONJ", "ADP", "NUM", "PART"]
]

# 2. Starting row & column. The first key on the palette starts at the position (start_row, start_column)
start_row = 3
start_column = 1

# 3. Max number of rows in one column
max_row = 8

# 4. The key type
type = "ActionBmwKey"
# End of configurable options

final_json = {
    "name": "BMW Palette",
    "cells": {}
}
current_row = start_row - 1
current_column = start_column - 1

with open(source_json_file, 'r') as file:
    source_data = json.load(file)

count = 0
for pos in pos_in_order:
    is_fresh_start = True
    if isinstance(pos, list):
        for subpos in pos:
            for one_symbol in source_data[subpos]:
                (current_row, current_column) = incrementPosition(True if is_fresh_start else False, current_row, current_column, start_row, max_row)
                is_fresh_start = False
                write_to_json(final_json, one_symbol, type, current_row, current_column)
                count = count + 1
    else:
        for one_symbol in source_data[pos]:
            (current_row, current_column) = incrementPosition(True if is_fresh_start else False, current_row, current_column, start_row, max_row)
            is_fresh_start = False
            write_to_json(final_json, one_symbol, type, current_row, current_column)
            count = count + 1

# Write the JSON into a file
with open(output_json_file, "w") as json_file:
    json_file.write(json.dumps(final_json, indent=4))
print(f"The final JSON is written into {output_json_file}")
print(f"Total number of symbols: {count}")
