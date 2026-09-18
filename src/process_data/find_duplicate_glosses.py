# python find_duplicate_glosses.py <input_cleaned_dict_json_path> <duplicate_glosses_dict_json_path>
# python find_duplicate_glosses.py ../data/bliss_dict/bliss_dict_multi_langs.json ../data/bliss_dict/duplicate_glosses.json

# This script processes a JSON file of Blissymbols dictionary in gloss to identify
# duplicate glosses across different items, taking into account their associated
# metadata such as 'is_old' status and 'semantics'. The output is a JSON
# structure that groups duplicate glosses by language, along with the IDs of
# the items that share those glosses.

# ========================================
# SUMMARY REPORT
# Total duplicate groups found: 5170
# ----------------------------------------
# Language   | Groups Found
# ----------------------------------------
# af         | 481
# de         | 632
# en         | 305
# es         | 374
# fi         | 441
# fr         | 249
# hu         | 392
# lv         | 570
# nl         | 371
# no         | 360
# po         | 348
# ru         | 341
# sv         | 306
# ========================================

import json
import argparse
import sys


def get_meta_signature(item):
    """
    Extracts the 'is_old' flag and 'semantics' object to create a
    hashable signature for comparison.
    Returns a tuple: (is_old (bool), semantics_json_string)
    """
    is_old = item.get("is_old", False)
    semantics = item.get("semantics", {})
    # Dump semantics to string with sorted keys to ensure consistent hashing
    semantics_str = json.dumps(semantics, sort_keys=True)
    return (is_old, semantics_str)


def format_key(gloss, is_old, semantics_str):
    """
    Formats the JSON key. If metadata exists, appends it to the gloss
    to distinguish it from other usages of the same word.
    """
    extras = []

    if is_old:
        extras.append("OLD")

    semantics = json.loads(semantics_str)
    if semantics:
        # Format semantics nicely, e.g., "POS: noun"
        sem_parts = [f"{k}: {v}" for k, v in semantics.items()]
        extras.extend(sem_parts)

    if not extras:
        return gloss

    return f"{gloss} ({', '.join(extras)})"


def find_duplicates(input_path, output_path):
    print(f"Reading {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Structure: { lang_code: { gloss_string: { meta_signature: [list_of_ids] } } }
    grouped_data = {}

    print("Analyzing glosses and metadata...")

    # Handle dictionary input (keyed by ID)
    items = [(item_id, item) for item_id, item in data.items()]

    for item_id, item in items:
        glosses = item.get("glosses", {})

        # Get the comparison signature for this specific item
        # (is_old, "{'POS': 'noun'}")
        meta_sig = get_meta_signature(item)

        for lang, gloss_list in glosses.items():
            if lang not in grouped_data:
                grouped_data[lang] = {}

            for gloss_text in gloss_list:
                if gloss_text not in grouped_data[lang]:
                    grouped_data[lang][gloss_text] = {}

                if meta_sig not in grouped_data[lang][gloss_text]:
                    grouped_data[lang][gloss_text][meta_sig] = []

                grouped_data[lang][gloss_text][meta_sig].append(item_id)

    # Filter results and format output
    # Structure: { lang_code: { "gloss (meta)": [id1, id2] } }
    final_output = {}
    total_duplicate_count = 0
    stats = {}

    for lang, gloss_map in grouped_data.items():
        lang_results = {}

        for gloss_text, meta_groups in gloss_map.items():
            for meta_sig, id_list in meta_groups.items():

                # We only want glosses with MORE THAN ONE ID sharing the exact same metadata
                if len(id_list) > 1:
                    is_old, semantics_str = meta_sig

                    # Create a unique key for the JSON output.
                    key_string = format_key(gloss_text, is_old, semantics_str)

                    # Sort IDs numerically if possible for cleaner output
                    sorted_ids = sorted(id_list, key=lambda x: int(x) if x.isdigit() else x)
                    lang_results[key_string] = sorted_ids
                    total_duplicate_count += 1

        if lang_results:
            # Sort keys alphabetically
            final_output[lang] = dict(sorted(lang_results.items()))
            stats[lang] = len(lang_results)

    # --- REPORTING SECTION ---
    print("\n" + "="*40)
    print("SUMMARY REPORT")
    print(f"Total duplicate groups found: {total_duplicate_count}")
    print("-" * 40)
    print(f"{'Language':<10} | {'Groups Found':<15}")
    print("-" * 40)

    for lang in sorted(stats.keys()):
        print(f"{lang:<10} | {stats[lang]:<15}")
    print("="*40 + "\n")
    # -------------------------

    print(f"Writing output to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print("Done.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


parser = argparse.ArgumentParser(description="Find duplicate Blissymbols sharing the same gloss and semantics.")
parser.add_argument("json_in", help="Location of the cleaned items JSON file")
parser.add_argument("json_out", help="Location of the output JSON report")

args = parser.parse_args()

find_duplicates(args.json_in, args.json_out)
