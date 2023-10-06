'''
Read the "encoding_codes" section from bmw.json, loop through every symbol to print the POS
of every word in every symbol.

The output JSON structure:
{
    "noun": ["a", "b"...],
    "verb": ["a", "b"...],
    ...
}
Items in every POS array are sorted in alphabetic order.

Example: python sort_by_pos.py ../data/bmw.json ../data/intermediate_BMW_conversion_data/symbols_in_pos.json
'''

import json
import sys
import spacy
import re

source_json_file = sys.argv[1]
output_json_file = sys.argv[2]

final_json = {}
all_pos = []
remove_special_chars = True

with open(source_json_file, 'r') as file:
    data = json.load(file)
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    for symbol, bci_av_id in data["encoding_codes"].items():
        if remove_special_chars:
            pattern = r'[^a-zA-Z0-9\s]'  # This pattern will keep letters, digits, and spaces
            cleaned_symbol = re.sub(pattern, '', symbol)

        doc = nlp(cleaned_symbol)
        for token in doc:
            if token.pos_ not in final_json:
                final_json[token.pos_] = []
            final_json[token.pos_].append(symbol)
            break

# sort each array in alphabetic order
symbols_count = 0
for pos, itemArray in final_json.items():
    sorted = itemArray.sort()
    symbols_count = symbols_count + len(itemArray)

# add BCI-AV-ID for every item
for pos, itemArray in final_json.items():
    array_with_bci_av_id = []
    for item in itemArray:
        array_with_bci_av_id.append({item: data["encoding_codes"][item]})
    final_json[pos] = array_with_bci_av_id

final_json["all_pos"] = list(final_json.keys())


# Write the JSON into a file
with open(output_json_file, "w") as json_file:
    json_file.write(json.dumps(final_json, indent=4))
print(f"The final JSON is written into {output_json_file}")

print(f"Total number of symbols: {symbols_count}")
