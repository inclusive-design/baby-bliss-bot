# python convert_dataset_eng_to_bliss.py <action_indicator_id_pairs_json> <input_eng_dataset_py> <output_bliss_dataset_py>
# python convert_dataset_eng_to_bliss.py ./data/action_indicator_id_pairs.json ./data/dataset_action_indicator_eng.py ./data/dataset_action_indicator_bliss.py

# This script converts an English-language dataset in a python list into a Blissymbolics-compatible format using
# predefined noun and verb mappings defined in a JSON file (<action_indicator_id_pairs_json>).

import sys
import json
import re

if len(sys.argv) != 4:
    print("Usage: python convert_dataset_eng_to_bliss.py <action_indicator_id_pairs_json> <input_eng_dataset_py> <output_bliss_dataset_py>")
    sys.exit(1)

action_indicator_id_pairs_json = sys.argv[1]
input_eng_dataset_py = sys.argv[2]
output_bliss_dataset_py = sys.argv[3]


def format_verb_composition(composition):
    """Convert composition array to BLISS format."""
    result = []
    for item in composition:
        if item == ";":
            continue
        elif item == 8993:
            result.insert(0, "[BLISS_8993]")
        else:
            result.append(f"[BLISS_{item}]")
    return "".join(result)


def load_pairs_data(action_indicator_id_pairs_json):
    """Load and process the pairs.json file to create lookup dictionaries."""
    pairs_data = json.load(open(action_indicator_id_pairs_json, "r"))

    # Create lookup dictionaries
    noun_lookup = {}  # noun_only -> id
    verb_lookup = {}  # infinitive_form_verbs -> composition

    for pair in pairs_data:
        # Process noun data
        if "noun" in pair and "noun_only" in pair["noun"]:
            noun_only = pair["noun"]["noun_only"].lower()
            noun_id = pair["noun"]["id"]
            noun_lookup[noun_only] = f"[BLISS_{str(noun_id)}]"

        # Process verb data
        if "verb-(to)" in pair and "infinitive_form_verbs" in pair["verb-(to)"]:
            verb_forms = pair["verb-(to)"]["infinitive_form_verbs"]
            composition = pair["verb-(to)"]["composition"]

            # Split by comma and clean up each verb form
            for verb_form in verb_forms:
                verb_lookup[verb_form.lower()] = format_verb_composition(composition)

    return noun_lookup, verb_lookup


def process_sentence(sentence, noun_lookup, noun_replace_pattern, verb_lookup, verb_replace_pattern):
    """Process a single sentence according to the rules."""
    # Replace single quotes
    sentence = sentence.replace("’", "'")

    # Return comment lines unchanged
    if sentence.strip().startswith("#"):
        return sentence

    # Verbs and nouns replacements
    def make_replacer(lookup):
        """Create a replacement function for a specific lookup table."""
        def replace_func(match):
            # Get the matched word (group 2)
            matched_text = match.group(2)

            # Find the key in lookup that matches (case-insensitive)
            for key, value in lookup.items():
                if str(key).lower() == matched_text.lower():
                    return value
            return match.group(0)  # Return the full match if no replacement found
        return replace_func

    # Perform the replacement
    processed_sentence = re.sub(verb_replace_pattern, make_replacer(verb_lookup), sentence, flags=re.IGNORECASE)
    processed_sentence = re.sub(noun_replace_pattern, make_replacer(noun_lookup), processed_sentence, flags=re.IGNORECASE)

    return processed_sentence


def get_lookup_pattern(lookup_table):
    """Create a regex pattern for the lookup table."""
    return r'(\s?)\b(' + '|'.join(re.escape(key) for key in lookup_table.keys()) + r')\b'


noun_lookup, verb_lookup = load_pairs_data(action_indicator_id_pairs_json)
noun_replace_pattern = get_lookup_pattern(noun_lookup)
verb_replace_pattern = get_lookup_pattern(verb_lookup)

# Read the dataset file
with open(input_eng_dataset_py, "r", encoding="utf-8") as f:
    content = f.read()

# Process line by line while preserving structure.
# Return comments and empty lines unchanged. Only process sentences.
lines = content.split("\n")
processed_lines = []

for line in lines:
    if line.strip().startswith("#") or line.strip().startswith("\""):
        # Process sentences and comments
        processed_line = process_sentence(line, noun_lookup, noun_replace_pattern, verb_lookup, verb_replace_pattern)
        processed_lines.append(processed_line)
    else:
        # Keep other lines as is (variable declarations, brackets, etc.)
        processed_lines.append(line)

# Write output
with open(output_bliss_dataset_py, "w", encoding="utf-8") as f:
    # Replace the variable name in the first line if it exists
    output_content = "\n".join(processed_lines)
    output_content = output_content.replace("data_for_action_indicator", "dataset_action_indicator_bliss")
    f.write(output_content)

print(f"Completed. Output written to {output_bliss_dataset_py}")
