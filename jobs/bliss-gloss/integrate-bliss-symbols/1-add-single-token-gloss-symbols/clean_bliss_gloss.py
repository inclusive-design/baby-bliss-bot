# python clean_bliss_gloss.py <input_gloss_file> <output_json_file> <shared_gloss_output_file>
# python clean_bliss_gloss.py ../../data/bliss_symbol_explanations.json ./data/bliss_gloss_cleaned.json ./data/bliss_gloss_shared.json

# This script transforms an input JSON array of objects (each with an id and description) into a
# structured output JSON object that maps each id to a cleaned and processed list of gloss strings.
#
# The script performs the following processing steps in order:
# 1. Handle special IDs in a predefined list for punctuation, numbers, single letters etc. These IDs
# are assigned with predefined glosses;
# 2. Replaces all underscores (_) with spaces;
# 3. Remove suffix "(OLD)" to support legacy symbols.
# 4. Handle plurals "(s)" by splitting the corresponding gloss ("gloss(s)") into a singular form
# ("glove") and a plural form ("gloves").
# 5. Process infinitive verb marker "-(to)" by prepending "to " to every gloss for that entry
# (e.g., "advance,go-(to)" becomes "to advance", "to go").
# 6. Extract and retain parenthetical suffixes ("autumn, fall (ckb)") by appending them to each gloss
# ("autumn (ckb)", "fall (ckb)").

# Input Gloss JSON (`<input_gloss_file>`):
# [
#   {
#     "id": "13174",
#     "description": "chocolate_spread_(OLD)"
#   },
#   {
#     "id": "12322",
#     "description": "ability,capability,capacity,potential"
#   }
#  ...
# ]
#
# Output JSON File Structure (`<output_json_file>`):
# The output JSON file is a dictionary where each key is a Bliss symbol ID, and the value is a list of cleaned glosses.
# Example:
# ```json
# {
#   "13174": ["chocolate spread"],
#   "12322": ["ability", "capability", "capacity", "potential"]
#   ...
# }
# ```

import os
import sys
import json
import re
from collections import defaultdict

if len(sys.argv) != 4:
    print("Usage: python clean_bliss_gloss.py <input_gloss_file> <output_json_file> <shared_gloss_output_file>")
    sys.exit(1)

input_gloss_file = sys.argv[1]
output_json_file = sys.argv[2]
shared_gloss_output_file = sys.argv[3]

# Predefined special glosses for specific IDs. These glosses bypass normal processing
# and are directly assigned.
special_glosses = {
    "8483": ["!"], "8484": ["%"], "8485": ["?"], "8486": ["."],
    "8487": [","], "8488": [":"], "8489": ["'"], "8490": ["degree"],
    "8496": ["0"], "8497": ["1"], "8498": ["2"], "8499": ["3"],
    "8500": ["4"], "8501": ["5"], "8502": ["6"], "8503": ["7"],
    "8504": ["8"], "8505": ["9"], "8521": ["a"], "8522": ["b"],
    "8523": ["c"], "8524": ["d"], "8525": ["e"], "8526": ["f"],
    "8527": ["g"], "8528": ["h"], "8529": ["i"], "8530": ["j"],
    "8531": ["k"], "8532": ["l"], "8533": ["m"], "8534": ["n"],
    "8535": ["o"], "8536": ["p"], "8537": ["q"], "8538": ["r"],
    "8539": ["s"], "8540": ["t"], "8541": ["u"], "8542": ["v"],
    "8543": ["w"], "8544": ["x"], "8545": ["y"], "8546": ["z"],
    "8551": ["A"], "8552": ["B"], "8553": ["C"], "8554": ["D"],
    "8555": ["E"], "8556": ["F"], "8557": ["G"], "8558": ["H"],
    "8559": ["I"], "8560": ["J"], "8561": ["K"], "8562": ["L"],
    "8563": ["M"], "8564": ["N"], "8565": ["O"], "8566": ["P"],
    "8567": ["Q"], "8568": ["R"], "8569": ["S"], "8570": ["T"],
    "8571": ["U"], "8572": ["V"], "8573": ["W"], "8574": ["X"],
    "8575": ["Y"], "8576": ["Z"]
}

if not os.path.exists(input_gloss_file):
    print(f"Error: Input gloss file '{input_gloss_file}' does not exist.")
    sys.exit(1)

try:
    with open(input_gloss_file, 'r') as infile:
        data = json.load(infile)

    primary_output_data = {}
    gloss_to_ids_map = defaultdict(list)

    for item in data:
        item_id = item.get("id")
        is_old = False

        # If the ID is in special_glosses, use its value directly and skip processing.
        if item_id in special_glosses:
            final_glosses = special_glosses[item_id]
        else:
            description = item.get("description", "")
            if not description:
                continue

            # 1. Replace "_" with " "
            processed_desc = description.replace("_", " ")

            # 2. Remove "(OLD)" suffix and any preceding space/underscore
            if processed_desc.endswith("(OLD)"):
                processed_desc = processed_desc[:-5].rstrip()
                is_old = True

            # 3. Extract parenthetical suffix (e.g., "(ckb)")
            parenthetical_suffix = ""
            # This regex finds a space followed by parentheses at the end of the string
            match = re.search(r'\s(\([^)]+\))$', processed_desc)
            if match:
                parenthetical_suffix = match.group(1)
                # Remove the matched suffix from the description
                processed_desc = processed_desc[:match.start()].strip()

            # 4. Handle "-(to)" suffix and prepend "to"
            add_to_prefix = False
            if processed_desc.endswith("-(to)"):
                processed_desc = processed_desc[:-5]
                add_to_prefix = True

            # 5. Split by comma for other cases
            initial_glosses = processed_desc.split(',')

            # 6. Expand glosses containing "(s)"
            expanded_glosses = []
            for gloss in initial_glosses:
                stripped_gloss = gloss.strip()
                if '(s)' in stripped_gloss:
                    # Add singular form (replace '(s)' with '')
                    expanded_glosses.append(stripped_gloss.replace('(s)', '').strip())
                    # Add plural form (replace '(s)' with 's')
                    expanded_glosses.append(stripped_gloss.replace('(s)', 's').strip())
                else:
                    expanded_glosses.append(stripped_gloss)

                # Append the extracted suffix to all glosses
                if parenthetical_suffix:
                    suffixed_glosses = [f"{g} {parenthetical_suffix}" for g in expanded_glosses if g]
                else:
                    suffixed_glosses = [g for g in expanded_glosses if g]

                # Prepend "to " if needed
                if add_to_prefix:
                    final_glosses = [f"to {g}" for g in suffixed_glosses if g]
                else:
                    final_glosses = [g for g in suffixed_glosses if g]

        new_item = {
            "glosses": final_glosses
        }

        for key, value in item.items():
            if key != "id" and key != "description":
                new_item[key] = value
        if is_old:
            new_item["is_old"] = True

        primary_output_data[item_id] = new_item
        # Populate the map for tracking shared glosses
        for gloss in final_glosses:
            gloss_to_ids_map[gloss].append(item_id)

    with open(output_json_file, 'w') as outfile:
        json.dump(primary_output_data, outfile, indent=2)
    print(f"Successfully processed generated {output_json_file} with total {len(primary_output_data)} entries.")

    # Find shared glosses and write the second output file
    shared_glosses_data = []
    for gloss, ids in gloss_to_ids_map.items():
        if len(ids) > 1:
            shared_glosses_data.append({"gloss": gloss, "ids": sorted(ids)})

    # Sort the final list alphabetically by gloss for consistent output
    shared_glosses_data.sort(key=lambda x: x['gloss'])

    with open(shared_gloss_output_file, 'w') as outfile:
        json.dump(shared_glosses_data, outfile, indent=2)

    print(f"Successfully identified shared glosses and generated: {shared_gloss_output_file}. Total shared glosses: {len(shared_glosses_data)}.")

except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{input_gloss_file}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
