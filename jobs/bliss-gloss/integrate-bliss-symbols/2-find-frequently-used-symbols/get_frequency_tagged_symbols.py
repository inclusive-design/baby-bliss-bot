# python get_frequency_tagged_symbols.py <frequency_tagged_symbols_csv> <output_tagged_symbols_json>
# python get_frequency_tagged_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
import csv
import json
import sys

if len(sys.argv) != 3:
    print("Usage: python get_frequency_tagged_symbols.py <frequency_tagged_symbols_csv> <output_tagged_symbols_json>")
    sys.exit(1)

input_csv = sys.argv[1]
output_json = sys.argv[2]

# Initialize the result structure with keys "1" to "5" and "None"
result = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "None": []
}

with open(input_csv, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        if not row:  # Skip empty rows
            continue

        bci_av_id = int(row[0].strip())
        tag_str = row[1].strip()
        tag_str = tag_str if tag_str else "None"

        result[tag_str].append(bci_av_id)

# Write the result to JSON
with open(output_json, "w") as jsonfile:
    json.dump(result, jsonfile, indent=2)
    print(f"Output written to {output_json}")
