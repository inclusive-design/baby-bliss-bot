'''
Loops through bmw.json, find all encoding icons, find their BCI-AV-IDs, then populate
"encoding_symbols" section in the bmw.json

Usage: python populate_encoding_symbols.py source_bmw_path bliss_explanation_json_location output_bmw_location
Parameters:
  source_bmw_path: The path where bmw.json is
  bliss_explanation_json_location: The location of the JSON file that contains the translation between Bliss
  BCI-AV-ID and its language translation
  output_bmw_location: The location of the output filename for bmw.json structure. If it doesn't exist, the
  script will create it
Return: None

Example: python populate_encoding_symbols.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
'''

import json
import sys

# Find the Bliss id for the given text:
# 1. Search through the given map first. If found, return id. Otherwise, continue the next step;
# 2. Search through the explanation JSON file. If found, return id. Otherwise, return None.
def get_bliss_id(text, default_encoding_symbol, explanation_json_data):
    text = text.lower()

    for key, value in default_encoding_symbol.items():
        if text == key.lower():
            return value

    for item in explanation_json_data:
        defs = item["description"].lower().split(',')

        # Skip old descriptions
        if defs[-1].endswith("_(OLD)"):
            continue
        # The suffix "-(to)" indicates a verb. It's not part of the definition
        if defs[-1].endswith("-(to)"):
            defs[-1] = defs[-1][:-6]
        if text in defs:
            return int(item["id"])

    return None


source_json_file = sys.argv[1]
bliss_explanation_json_location = sys.argv[2]
output_bmw_location = sys.argv[3]

default_encoding_symbol = {
    "ABS TIME": ["HC8N:0,8;S4:2,10", "/", 17732],
    "ADJ.": 25554,
    "ADJ.+ER": 24879,
    "ADJ.+EST": 24944,
    "AM/BE": 12639,
    "COMMUN.": 13390,
    "CONJ.": 23409,
    "DEM.": [17720, "/", 17697],
    "DIRECTN": 13682,
    "EXCLAIM": 14947,
    "FORCE": 22914,
    "FURNITUR": 14416,
    "HAVE": 14685,
    "HE+": 14687,
    "I+": 14916,
    "INTERJ.": [24961, "/", 15947],
    "IT+": 14960,
    "LANG.": 15172,
    "LEGSFEET": 15189,
    "LIMITIME": 15212,
    "MINSPEAK": [14647, "/", 15972, "/", 15172],
    "N PERSON": 16161,
    "N.PER PL": [16161, ";", 9011],
    "NOUN PL.": [17717, ";", 9011],
    "OUTWORL": 15411,
    "OUTWORLD": 15411,
    "PREP.": [12324, "/", 17717],
    "PREVERB": [13867, "/", 12335],
    "PREVRB+S": [13867, "/", 12335],
    "POSS.": 24925,
    "RECEIVE": 14435,
    "THEY+": 17714,
    "TO+VERB": 13860,
    "YOU+": 18465,
    "SHE+": 14688,
    "SEXUAL": 16933,
    "VERB": 12335,
    "VERB+S": [12335, "/", 8499],
    "VERB+ED": [12335, "/", 15975],
    "VERB+EN": [12335, "/", 13949],
    "VERB+ING": [12335, "/", 14390],
    "WE+": 18212,
    "WRITE": 18285,
    "YES": 18294
}

# load bliss translation json file
with open(bliss_explanation_json_location, 'r') as file:
    bliss_explanation_json = json.load(file)

final_encoding_symbols = {}

with open(source_json_file, 'r') as file:
    data = json.load(file)

    unique_encoding_symbols = set()

    for message, value in data["encodings"].items():
        for item in value["encoding"]:
            unique_encoding_symbols.add(item)

    count = 0
    for item in unique_encoding_symbols:
        count = count + 1
        bliss_id = get_bliss_id(item, default_encoding_symbol, bliss_explanation_json)
        if bliss_id is None:
            print("Error: cannot find bliss id for '{item}'")
        else:
            print(f"{item}: {bliss_id}")
            final_encoding_symbols[item] = bliss_id

    data["encoding_symbols"] = final_encoding_symbols

# Write the JSON into a file
with open(output_bmw_location, "w") as json_file:
    json_file.write(json.dumps(data, indent=4))

print(f"Total: {count}")
