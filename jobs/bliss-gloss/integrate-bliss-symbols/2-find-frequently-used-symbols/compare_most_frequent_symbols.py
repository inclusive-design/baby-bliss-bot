# python compare_most_frequent_symbols.py <bliss_symbol_explanations_json> <output_overlapped_symbols_json> <output_only_in_standard_board_json> <output_only_in_tag1_json>
# python compare_most_frequent_symbols.py ../../data/bliss_symbol_explanations.json ./output/overlapped_symbols.json ./output/only_in_standard_board.json ./output/only_in_tag1.json

import json
import sys

if len(sys.argv) != 5:
    print("Usage: python compare_most_frequent_symbols.py <bliss_symbol_explanations_json> <output_overlapped_symbols_json> <output_only_in_standard_board_json> <output_only_in_tag1_json>")
    sys.exit(1)

bliss_symbol_explanations_json = sys.argv[1]
output_overlapped_symbols_json = sys.argv[2]
output_only_in_standard_board_json = sys.argv[3]
output_only_in_tag1_json = sys.argv[4]

standard_board_symbols_json = "./output/standard_board_symbols.json"
frequency_tagged_symbols_json = "./output/frequency_tagged_symbols.json"

explanations_data = json.load(open(bliss_symbol_explanations_json, "r"))
standard_board_data = json.load(open(standard_board_symbols_json, "r"))
frequency_tagged_ids = json.load(open(frequency_tagged_symbols_json, "r"))["1"]

standard_board_ids = {int(item["id"]) for item in standard_board_data}
symbol_explanation_id_lookup = {int(item["id"]): item for item in explanations_data}

# IDs in both lists
overlapping = [x for x in standard_board_data if int(x["id"]) in frequency_tagged_ids]

# IDs only in the standard board
only_in_standard_board = [x for x in standard_board_data if int(x["id"]) not in frequency_tagged_ids]

# IDs only in the frequency tagged symbols (tag1)
only_in_tag1 = []

for bliss_id in frequency_tagged_ids:
    if bliss_id not in standard_board_ids:
        if bliss_id in symbol_explanation_id_lookup:
            match = symbol_explanation_id_lookup[bliss_id]
            only_in_tag1.append({
                "id": bliss_id,
                "description": match["description"],
                "explanation": match["explanation"]
            })

print(f"Standard board symbols: {len(standard_board_data)}")
print(f"Standard board ids: {len(standard_board_ids)}")
print(f"Frequency tagged symbols: {len(frequency_tagged_ids)}")

# Write the result to JSON
with open(output_overlapped_symbols_json, "w") as jsonfile:
    json.dump(overlapping, jsonfile, indent=2)
    print(f"{len(overlapping)} Overlapping symbols written to {output_overlapped_symbols_json}")

with open(output_only_in_standard_board_json, "w") as jsonfile:
    json.dump(only_in_standard_board, jsonfile, indent=2)
    print(f"{len(only_in_standard_board)} Only in standard board symbols written to {output_only_in_standard_board_json}")

with open(output_only_in_tag1_json, "w") as jsonfile:
    json.dump(only_in_tag1, jsonfile, indent=2)
    print(f"{len(only_in_tag1)} Only in tag1 symbols written to {output_only_in_tag1_json}")
