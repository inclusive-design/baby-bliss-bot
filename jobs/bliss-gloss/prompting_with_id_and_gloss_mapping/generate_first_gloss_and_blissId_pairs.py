# This script reads in ../data/bliss_gloss_cleaned.json, iterates through each record, and generates
# an output that every record is converted to a mapping of `gloss[0]=blissID`

# python generate_first_gloss_and_blissId_pairs.py ../data/bliss_gloss_cleaned.json unique.txt duplicate.txt

import json
import sys
from collections import defaultdict


# Process the gloss data and generate the mappings
def process_json_mappings(gloss_dict):
    # Dictionary to track first glosses and their associated Bliss IDs
    gloss_to_blissId = defaultdict(list)

    # Process each record
    for blissId, glosses in gloss_dict.items():
        if glosses:  # Check if the gloss array is not empty
            first_gloss = glosses[0]
            # Track all glosses and their Bliss IDs for duplicate checking
            gloss_to_blissId[first_gloss].append(blissId)

    # Separate mappings into unique and duplicates
    unique_equations = []
    duplicate_mappings = []
    duplicates_report = []

    for first_gloss, bliss_ids in gloss_to_blissId.items():
        if len(bliss_ids) == 1:
            # Unique mapping
            unique_equations.append(f'"{first_gloss}"={bliss_ids[0]}')
        else:
            # Handle duplicates
            unique_equations.append(f'"{first_gloss}"={bliss_ids[0]}')  # Use the first Bliss ID
            for bliss_id in bliss_ids[1:]:
                duplicate_mappings.append(f'"{first_gloss}"={bliss_id}')
            duplicates_report.append(
                f'"{first_gloss}" is mapped to multiple Bliss IDs: {", ".join(bliss_ids)}'
            )

    return unique_equations, duplicate_mappings, duplicates_report


# Validate input arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <input_json_file> <unique_output_file> <duplicates_output_file>")
    sys.exit(1)

input_gloss_file = sys.argv[1]
unique_output_file = sys.argv[2]
duplicates_output_file = sys.argv[3]

# Read the JSON file
with open(input_gloss_file, 'r') as file:
    gloss_dict = json.load(file)

# Process the data
unique_equations, duplicate_mappings, duplicates_report = process_json_mappings(gloss_dict)

# Write unique mappings to the first file
with open(unique_output_file, 'w') as file:
    file.write("Unique Mappings:\n")
    file.write("\n".join(unique_equations))

# Write duplicates and unused mappings to the second file
with open(duplicates_output_file, 'w') as file:
    file.write("Mappings Excluded Due to Duplicates:\n")
    file.write("\n".join(duplicate_mappings))
    file.write("\n\nDuplicate Report:\n")
    file.write("\n".join(duplicates_report))

print("Processing complete!")
print(f"Unique mappings written to: {unique_output_file}")
print(f"Duplicates and report written to: {duplicates_output_file}")
