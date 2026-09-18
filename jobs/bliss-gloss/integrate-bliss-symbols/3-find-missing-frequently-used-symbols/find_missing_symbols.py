# python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json>
# python find_missing_symbols.py ../1-add-single-token-gloss-symbols/output/bliss_ids_added.json ../2-find-frequently-used-symbols/output/overlapped_symbols.json ./output/missing_symbols.json

import json
import sys

if len(sys.argv) != 4:
    print("Usage: python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json> <output_missing_symbols_json>")
    sys.exit(1)

symbols_in_model_json = sys.argv[1]
most_common_symbols_json = sys.argv[2]
output_missing_symbols_json = sys.argv[3]

symbols_in_model = json.load(open(symbols_in_model_json, 'r'))
existing_symbol_ids = set(map(int, symbols_in_model.keys()))

most_common_symbols = json.load(open(most_common_symbols_json, 'r'))
most_common_symbol_ids = {int(item["id"]): item for item in most_common_symbols}

# Find missing elements
missing_ids = [bci_av_id for bci_av_id in most_common_symbol_ids if bci_av_id not in existing_symbol_ids]

# Output the result
with open(output_missing_symbols_json, 'w') as f:
    json.dump(missing_ids, f, indent=2)
    print(f"{len(missing_ids)} missing symbol IDs saved to {output_missing_symbols_json}")
