# python expand_bliss_dict.py json_in tsv_in json_out
# python expand_bliss_dict.py ../../jobs/bliss-gloss/data/bliss_symbol_explanations.json ../data/bliss_dict/BCI-AV_SKOG_2025-02-15_multi-langs.tsv ../data/bliss_dict/bliss_symbol_explanations_multi_langs.json

# Expand Blissymbolics JSON dictionary with multilingual descriptions from TSV.
import json
import csv
import argparse
import sys


def get_column_indices(headers):
    """
    Maps the specific TSV headers to the target language codes requested.
    Returns a dict: { "lang_code": column_index }
    """

    # Mapping from TSV Header string -> Target JSON Key
    # Based on the TSV snippet and the requested code mapping provided.
    header_map = {
        "English": "en",
        "Swedish": "sv",
        "Norwegian": "no",
        "Finnish": "fi",
        "Hungarian": "hu",
        "German": "de",
        "Dutch": "nl",
        "Afrikaans": "af",
        "Russian": "ru",
        "Latvian": "lv",
        "Polish": "po",
        "French": "fr",
        "Spanish": "es",
        "Portugese - draft": "pt",  # Matches TSV snippet (typo included)
        "Portuguese - draft": "pt",  # handling potential correct spelling
        "Italian - draft": "it",
        "Danish - draft": "dk"
    }

    col_indices = {}
    id_index = -1

    for idx, header in enumerate(headers):
        clean_header = header.strip()

        # Identify the ID column (first occurrence of BCI-AV#)
        if clean_header == "BCI-AV#" and id_index == -1:
            id_index = idx
            continue

        # Identify Language columns
        if clean_header in header_map:
            lang_code = header_map[clean_header]
            col_indices[lang_code] = idx

    return id_index, col_indices


def load_translations(tsv_path):
    """
    Reads the TSV file and returns a dictionary mapping IDs to translation objects.
    Format: { "id_string": { "en": "...", "sv": "..." } }
    """
    translations = {}

    try:
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')

            try:
                headers = next(reader)
            except StopIteration:
                print("Error: TSV file is empty.")
                sys.exit(1)

            id_index, col_indices = get_column_indices(headers)

            if id_index == -1:
                print("Error: Could not find 'BCI-AV#' column in TSV.")
                sys.exit(1)

            for row in reader:
                # Skip empty rows
                if not row:
                    continue

                # Safely get ID
                if id_index < len(row):
                    row_id = row[id_index].strip()

                    # Create the description object for this ID
                    desc_obj = {}
                    for lang_code, col_idx in col_indices.items():
                        if col_idx < len(row):
                            # Add value, stripped of whitespace
                            desc_obj[lang_code] = row[col_idx].strip()
                        else:
                            desc_obj[lang_code] = ""

                    translations[row_id] = desc_obj

    except FileNotFoundError:
        print(f"Error: TSV file not found at {tsv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        sys.exit(1)

    return translations


def process_dictionary(json_in_path, tsv_in_path, json_out_path):
    # 1. Load Translations from TSV
    print(f"Loading translations from {tsv_in_path}...")
    translation_map = load_translations(tsv_in_path)
    print(f"Loaded translations for {len(translation_map)} items.")

    # 2. Load JSON Dictionary
    print(f"Loading JSON dictionary from {json_in_path}...")
    try:
        with open(json_in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_in_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

    # 3. Update Descriptions
    matched_count = 0
    for item in data:
        item_id = item.get("id")

        if item_id and item_id in translation_map:
            # Replace the description string/field with the new dictionary
            item["description"] = translation_map[item_id]
            matched_count += 1
        else:
            print(f"Warning: No translation found for ID {item_id}")

    print(f"Updated {matched_count} entries with multilingual descriptions.")

    # 4. Convert list to dictionary indexed by "id"
    print("Converting output to dictionary indexed by 'id'...")
    json_out = {}
    for item in data:
        item_id = item.get("id")
        if item_id:
            # Remove "id" from item since it's now the key
            item_copy = {k: v for k, v in item.items() if k != "id"}
            json_out[item_id] = item_copy
        else:
            print(f"Warning: Item has no 'id' field: {item}")

    # 5. Write Output
    print(f"Writing output to {json_out_path}...")
    try:
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(json_out, f, indent=2, ensure_ascii=False)
        print(f"Done. Wrote {len(json_out)} entries.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


parser = argparse.ArgumentParser(description="Expand Blissymbolics JSON dictionary with multilingual descriptions from TSV.")
parser.add_argument("json_in", help="Path to the input JSON dictionary file")
parser.add_argument("tsv_in", help="Path to the input TSV translation file")
parser.add_argument("json_out", help="Path to the output JSON file")

args = parser.parse_args()

process_dictionary(args.json_in, args.tsv_in, args.json_out)
