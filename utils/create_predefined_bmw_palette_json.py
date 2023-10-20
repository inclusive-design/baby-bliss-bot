'''
Create the JSON file for rendering BMW code keys on the BMW Palette in a pre-defined layout.

Example: python create_predefined_bmw_palette_json.py ../data/bmw.json ../data/bmw_palette.json
'''

import json
import sys
import uuid
from slugify import slugify


def add_to_json(final_json, one_code_label, type, bci_av_id, current_row, current_column):
    key = f"{slugify(one_code_label)}-{uuid.uuid4()}"
    print("key: ", key)
    final_json["cells"][key] = {
        "type": type,
        "options": {
            "label": one_code_label,
            "bciAvId": bci_av_id,
            "rowStart": current_row,
            "rowSpan": 1,
            "columnStart": current_column,
            "columnSpan": 1
        }
    }
    print(f"{one_code_label}: {current_row}, {current_column}")
    return final_json


bmw_json_file = sys.argv[1]
output_json_file = sys.argv[2]

# Configurable options
# 1. BMW code Keys are displayed based on their POS values. Each element in `pos_in_order` array
# means BMW code keys in this/these POS catgor(ies) start in a new column.
code_positions = [
    [None, None, "DEM.", "CONJ.", "ADVERB", "PREP.", "ADJ.", "ADJ.+ER", "ADJ.+EST", "NOUN", "NOUN PL.", "N PERSON", "N.PER PL", "ABS TIME"],
    ["OBJECT", "POSS.", "VERB", "VERB+S", "VERB+ING", "VERB+ED", "VERB+EN", "TO+VERB", "WHAT", "INTERJ.", "MINSPEAK", "FORCE", "NUMBER", "POEM"],
    ["I+", "WE+", "HELP", "GIVE", "COMMUN.", "TOOL", "COUNTRY", "LIMITIME", "YEAR", "MONTH", "DAY", "MANY", "PART", "RELATION"],
    ["YOU+", "THEY+", "RECEIVE", "WRITE", "EYE", "REPEAT", "WATER", "WEIGHT", "PUT", "FOOD", "WITHOUT", "ALSO", "EQUAL", "OPPOSITE"],
    ["HE+", "PREVERB", "QUIET", "WANT", "ELECTRIC", "RELIGION", "THINK", "YOUNG", "UMBRELLA", "IF", "OUTWORLD", "PAST", "PRESENT", "FUTURE"],
    ["SHE+", "PREVRB+S", "AM/BE", "STOP", "DIRECTN", "FORGIVE", "GO", "HAVE", "JOURNEY", "KING", "LOVE", "HOUSE", "RETURN", "LEGSFEET"],
    ["IT+", "NO", "DRESS", "ZEBRA", "EXCLAIM", "CAN", "VALUE", "BUT", "NEAR", "MAYBE", "MONEY", "FURNITUR", "YES", "SAY"],
    ["AIR", "OPEN", "NOT", "FINISH", "CAUSE", "BIRTH", "SEXUAL", "LANG."]
]

# 2. Starting row & column. The first key on the palette starts at the position (start_row, start_column)
start_row = 3
start_column = 1

# 4. The key type
type = "ActionBmwCodeCell"
# End of configurable options

final_json = {
    "name": "BMW Palette",
    "cells": {}
}

with open(bmw_json_file, 'r') as file:
    source_data = json.load(file)
    all_code_mapping = source_data["encoding_codes"]
    all_code_labels = list(all_code_mapping.keys())

total_number_code = 0
current_row = start_row
for one_row in code_positions:
    current_column = start_column

    for one_code_label in one_row:
        if one_code_label is None:
            current_column = current_column + 1
            continue
        else:
            if one_code_label in all_code_labels:
                add_to_json(final_json, one_code_label, type, all_code_mapping[one_code_label], current_row, current_column)
            else:
                print(f"Error: BCI-AV-ID is not found for {one_code_label}")
            current_column = current_column + 1
            total_number_code = total_number_code + 1
    current_row = current_row + 1

# Write the JSON into a file
with open(output_json_file, "w") as json_file:
    json_file.write(json.dumps(final_json, indent=4))
print(f"The final JSON is written into {output_json_file}")
print(f"Total number of codes: {total_number_code}")
