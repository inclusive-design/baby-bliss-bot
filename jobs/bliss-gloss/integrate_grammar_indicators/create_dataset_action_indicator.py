# python create_dataset_action_indicator.py <bliss_ids_in_model_json> <bliss_symbol_explanations_json> <output_id_pairs_json> <prompts_txt>")
# python create_dataset_action_indicator.py ./data/bliss_ids_added.json ../data/bliss_symbol_explanations.json ./data/action_indicator_id_pairs.json ./output/action_indicator_prompts.txt

# This script processes a Bliss symbol explanation JSON file to identify composite symbols
# formed by a noun combined with an action indicator, which converts the noun into the
# infinitive form of its corresponding verb.
#
# For each such composite symbol:
# - It ensures the base (noun) symbol is already present in the model.
#
# The script generates two output files:
# 1. A JSON file containing pairs of symbol IDs:
#    - The first ID is the base noun symbol.
#    - The second ID is the action indicator.
#    - Corresponding glosses for both symbols are included.
#
# 2. A text file containing prompts for training purposes:
#    - Each noun and each resulting infinitive verb have their own individual prompt lines.

import sys
import json

if len(sys.argv) != 5:
    print("Usage: python create_dataset_action_indicator.py <bliss_ids_in_model_json> <bliss_symbol_explanations_json> <output_id_pairs_json> <prompts_txt>")
    sys.exit(1)

bliss_ids_in_model_json = sys.argv[1]
bliss_symbol_explanations_json = sys.argv[2]
output_id_pairs_json = sys.argv[3]
prompts_txt = sys.argv[4]

bliss_ids_in_model = json.load(open(bliss_ids_in_model_json, "r"))
explanations_data = json.load(open(bliss_symbol_explanations_json, "r"))


# Convert noun formats by removing context information: "noun_(context)" to "noun"
def convert_none_format(text):
    separator_pos = text.find('_(')
    if separator_pos != -1:
        return text[:separator_pos].strip(), text[separator_pos + 2:len(text) - 1].replace("_", " ").strip()
    else:
        return text.strip(), None


# Convert verb formats like:
# 1. "verb1,verb2,verb3-(to)" to "to verb1,to verb2,to verb3"
# 2. "verb_(context)-(to)" to "to verb (context)"
def convert_verb_format(text):
    # Remove the "-(to)" suffix
    if text.endswith("-(to)"):
        text = text[:-5]  # Remove last 5 characters: "-(to)"

    # Split by comma and strip whitespace
    verbs = [verb.strip() for verb in text.split(",")]

    # Process each verb
    converted_verbs = []
    for verb in verbs:
        # Replace underscores with spaces, but handle parentheses specially
        if "(" in verb and ")" in verb:
            paren_start = verb.find("(")
            verb_part = verb[:paren_start].replace("_", " ").strip()
        else:
            # Just replace underscores with spaces
            verb_part = verb.replace("_", " ")

        converted_verbs.append(verb_part if verb_part == "can" else f"to {verb_part}")

    # Join with commas
    return converted_verbs


# Create a dictionary for quick lookup of composition items by id
composition_dict = {item["id"]: item for item in explanations_data}

result_list = []
prompts = []
count = 0
frequency_to_add_general_prompt = 20

for item in explanations_data:
    # print(f"Processing item: {item['id']} with composition: {'composition' in item}")
    # if ("composition" in item):
    #     print(f"length of composition: {len(item['composition'])}, {item['composition'][2] if len(item['composition']) > 2 else 'N/A'}")
    # Check if item has composition with length 3 and last element equals 8933
    if ("composition" in item and len(item["composition"]) == 3 and item["composition"][2] == 8993):
        # Get Bliss ID for the noun from composition
        noun_bliss_id = item["composition"][0]

        # Check if the noun id is already added to the model
        if str(noun_bliss_id) in bliss_ids_in_model:
            count += 1

            # Find the description of the first element in explanations_data
            if str(noun_bliss_id) in composition_dict:
                noun_bliss_id_description = composition_dict[str(noun_bliss_id)]["description"]
            else:
                print(f"Error: No description found for noun ID: {noun_bliss_id}")

            noun, context_for_noun = convert_none_format(noun_bliss_id_description)
            verbs = convert_verb_format(item["description"])

            if (count - 1) % frequency_to_add_general_prompt == 0:
                prompts.append("\nGenerate sentences using the following instructions:")

            prompts.append(f"Generate 3 sentences using the word \"{noun}\" as a noun.")
            prompts.append(f"Generate 3 sentences for each of the following infinitive forms: \"{','.join(verbs)}\".")

            if count % frequency_to_add_general_prompt == 0:
                prompts.append("\nThe output must be a valid Python list of strings, each representing one sentence, and assigned to the variable \"data_for_action_indicator\". Format your response as Python code only, with no additional explanation or comments.\n\n")

            # Create the result object
            result_obj = {
                "noun": {
                    "id": noun_bliss_id,
                    "description": noun_bliss_id_description,
                    "noun_only": noun,
                },
                "verb-(to)": {
                    "id": item["id"],
                    "composition": item["composition"],
                    "description": item["description"],
                    "infinitive_form_verbs": verbs,
                    "explanation": item["explanation"]
                }
            }

            result_list.append(result_obj)

with open(output_id_pairs_json, "w") as f:
    json.dump(result_list, f, indent=2)

print(f"{len(result_list)} pairs saved to {output_id_pairs_json}")

with open(prompts_txt, 'w') as file:
    for prompt in prompts:
        file.write(prompt + '\n')

print(f"{len(prompts)} prompts are saved to {prompts_txt}")
