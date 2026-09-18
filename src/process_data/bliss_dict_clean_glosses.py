# python bliss_dict_clean_glosses.py <input_bliss_explanation_json_path> <output_cleaned_dict_json_path>
# python bliss_dict_clean_glosses.py ../data/bliss_dict/bliss_symbol_explanations_multi_langs.json ../data/bliss_dict/bliss_dict_multi_langs.json

# This script processes a Blissymbolics dictionary JSON file to clean and standardize glosses,
# handle special cases, and build semantic mappings based on part-of-speech tags.

import json
import argparse
import sys
import re

# 1. Special IDs Configuration
SPECIAL_GLOSSES = {
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


def clean_single_gloss(text):
    """
    Handles expanding (s) into singular and plural forms.
    Returns a list of strings.
    """
    # Check for ending in (s), e.g., "glove(s)"
    # Use regex to find word ending in (s)
    match = re.search(r'^(.*)\(s\)$', text)
    if match:
        base = match.group(1)
        return [base, base + "s"]

    return [text]


def process_language_string(raw_text):
    """
    Takes a raw description string (e.g., "autumn, fall (ckb)")
    Returns a tuple: (list_of_cleaned_glosses, is_old_flag)
    """
    if not raw_text:
        return [], False

    text = raw_text
    is_old = False

    # 3. Remove suffix "_(OLD)" and flag it
    # We look for "_(OLD)" at the very end of the string
    if text.endswith("_(OLD)"):
        text = text[:-6]  # Remove last 6 chars
        is_old = True

    # 5. Remove suffix "-(to)"
    # Usually appears at the end of verbs like "run-(to)"
    if text.endswith("-(to)"):
        text = text[:-5]

    # 2. Replaces all underscores (_) with spaces
    text = text.replace("_", " ")

    # 6. Extract and retain parenthetical suffixes
    # Logic: If the string ends with a parenthetical group (like " (ckb)"),
    # that context applies to all comma-separated parts preceding it.
    # Regex: Look for any content, followed by a space and a parenthesized group at end of string.
    context_suffix = ""
    # Pattern: content group (non-greedy), followed by optional whitespace and (stuff) at end
    context_match = re.search(r'^(.*?)\s*(\(.*\))$', text)

    if context_match:
        # Check if the parenthesis is actually a context suffix or just part of a word
        # Heuristic: usually context suffixes apply to the whole csv list.
        # We separate the main text from the suffix.
        text = context_match.group(1)
        context_suffix = " " + context_match.group(2).strip()

    # Split by comma to get individual glosses
    parts = [p.strip() for p in text.split(',')]

    final_glosses = []
    for part in parts:
        if not part:
            continue

        # 4. Handle plurals "(s)"
        # This expands "word(s)" -> ["word", "words"]
        expanded = clean_single_gloss(part)

        # Re-attach context suffix if it existed
        for ex in expanded:
            final_glosses.append(ex + context_suffix)

    return final_glosses, is_old


def process_data(json_in_path, json_id_out_path):
    print(f"Reading {json_in_path}...")
    try:
        with open(json_in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Output containers
    id_map_output = {}

    print("Processing items...")
    # Handle dictionary input (keyed by ID)
    items = [(item_id, item) for item_id, item in data.items()]

    for item_id, item in items:
        description_obj = item.get("description", {})
        pos_val = item.get("pos")

        # Prepare new item structure
        new_item = item.copy()

        # Rename description -> glosses (will hold lists now)
        if "description" in new_item:
            del new_item["description"]

        new_glosses = {}
        item_is_old = False

        # Iterate through languages
        if isinstance(description_obj, dict):
            for lang, val in description_obj.items():

                # 1. Handle special IDs (only for English)
                if lang == "en" and item_id in SPECIAL_GLOSSES:
                    cleaned_list = SPECIAL_GLOSSES[item_id]
                else:
                    cleaned_list, found_old = process_language_string(val)
                    if found_old:
                        item_is_old = True

                if cleaned_list:
                    new_glosses[lang] = cleaned_list

        new_item["glosses"] = new_glosses

        # Add is_old flag if detected in any language (though usually en)
        if item_is_old:
            new_item["is_old"] = True

        # Handle Semantics based on POS
        semantics = {}
        if pos_val == "RED":
            semantics["POS"] = "verb"
        elif pos_val in ["YELLOW", "BLUE"]:
            semantics["POS"] = "noun"

        if "composition" in item and 9009 in item["composition"]:
            # Add concretization type shift
            semantics["TYPE_SHIFT"] = "concretization"

        if semantics:
            new_item["semantics"] = semantics

        id_map_output[item_id] = new_item

    # Write Output of ID Map
    print(f"Writing processed items to {json_id_out_path}...")
    try:
        with open(json_id_out_path, 'w', encoding='utf-8') as f:
            # Output as a dictionary keyed by ID
            json.dump(id_map_output, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing ID output: {e}")

    print("Done.")


parser = argparse.ArgumentParser(description="Process Blissymbolics dictionary: Clean glosses and build semantic maps.")
parser.add_argument("json_in", help="Location of the JSON with expanded dictionary")
parser.add_argument("json_id_out", help="Location of the output JSON (cleaned items)")

args = parser.parse_args()

process_data(args.json_in, args.json_id_out)
