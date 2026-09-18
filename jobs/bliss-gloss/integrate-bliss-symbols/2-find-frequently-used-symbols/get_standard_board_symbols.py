# python get_standard_board_symbols.py <bliss_standard_chart_json> <bliss_symbol_explanations_json> <output_symbols_file>")
# python get_standard_board_symbols.py ../../../../../adaptive-palette/public/palettes/bliss_standard_chart.json ../../data/bliss_symbol_explanations.json ./output/standard_board_symbols.json
# Note: <bliss_standard_chart_json> must point to the json file that contains other JSON files that the "branchTo" field refers to.

# This script processes a JSON file containing symbols on the Bliss standard board. The symbols in this file don't
# have their direct IDs but are represented by their composing symbols at the character level. Since this script needs
# to output symbols at the word-level, it compares this composing information with those in the Bliss symbol explanations
# JSON file, which contains the mapping from character-level composing information to symbol IDs.

import sys
import os
import json
from collections import deque

if len(sys.argv) != 4:
    print("Usage: python get_standard_board_symbols.py <bliss_standard_chart_json> <bliss_symbol_explanations_json> <output_symbols_file>")
    sys.exit(1)

bliss_standard_chart_json = sys.argv[1]
bliss_symbol_explanations_json = sys.argv[2]
output_file = sys.argv[3]

palette_directory = os.path.dirname(bliss_standard_chart_json)

processed = set()
queued = set()
queue = deque()
ids_with_composition = set()

# Start with the initial file and recursively find all referenced files led by the "branchTo" field
if os.path.isfile(bliss_standard_chart_json):
    queue.append(bliss_standard_chart_json)
    queued.add(bliss_standard_chart_json)
else:
    print(f"Error: Initial file {bliss_standard_chart_json} not found")
    sys.exit(1)

while queue:
    file_path = queue.popleft()
    queued.remove(file_path)
    processed.add(file_path)
    print(f"processing {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

    cells = data.get("cells", {})
    for cell_id, cell in cells.items():
        options = cell.get("options", {})
        branch_to = options.get("branchTo")
        bci_av_id = options.get("bciAvId")
        bci_av_id = (bci_av_id[0] if len(bci_av_id) == 1 else tuple(bci_av_id)) if isinstance(bci_av_id, list) else bci_av_id
        # print(f"Processing cell {cell_id} with branchTo: {branch_to} and bciAvId: {bci_av_id}")

        if bci_av_id is not None:
            ids_with_composition.add(bci_av_id)
            # print(f"Found bciAvId and pushed: {bci_av_id}")

        if branch_to is not None:
            # Process referenced file
            referenced_file = os.path.join(palette_directory, f"{branch_to}.json")
            if os.path.isfile(referenced_file):
                if referenced_file not in processed and referenced_file not in queued:
                    queue.append(referenced_file)
                    queued.add(referenced_file)
                    # print(f"Queued referenced file: {referenced_file}")
            else:
                print(f"Warning: Referenced file {branch_to}.json not found")

# Load the symbol explanations data and match on the IDs and compositions as needed
with open(bliss_symbol_explanations_json, "r") as f:
    explanations_data = json.load(f)

id_lookup = {}
composition_lookup = {}

for item in explanations_data:
    # Index by ID (convert to int for comparison)
    id_lookup[int(item["id"])] = item

    # Index by composition if it exists
    if "composition" in item:
        # Convert composition to tuple for hashable lookup
        comp_tuple = tuple(item["composition"])
        composition_lookup[comp_tuple] = item

results = []

# Process each item in ids_with_composition
for item in ids_with_composition:
    if isinstance(item, (int, str)):
        item_id = int(item)
        if item_id in id_lookup:
            match = id_lookup[item_id]
            results.append({
                "id": match["id"],
                "description": match["description"],
                "explanation": match["explanation"]
            })
        else:
            print(f"Error: No match found for ID: {item_id}")
    elif isinstance(item, tuple):
        # Handle tuple values
        if item in composition_lookup:
            match = composition_lookup[item]
            results.append({
                "id": match["id"],
                "description": match["description"],
                "explanation": match["explanation"]
            })
        else:
            print(f"Error: No match found for composition: {item}")
    else:
        print(f"Error: Unexpected item type: {type(item)} - {item}")

unique_results = []
seen_ids = set()

for item in results:
    if item["id"] not in seen_ids:
        seen_ids.add(item["id"])
        unique_results.append(item)

unique_results.sort(key=lambda x: int(x["id"]))

with open(output_file, "w") as f:
    json.dump(unique_results, f, indent=2)

print(f"\nTotal count of symbols: {len(unique_results)}")
print("Done! Results saved to", output_file)
