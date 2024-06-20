'''
python find_missing_encodings.py ../data/bmw.json ../data/intermediate_BMW_conversion_data/bmw_texts/ ../data/bliss_symbol_explanations.json missing_encodings.json error.txt
'''

import sys
import os
import json


# Joins elements starting at the position `start_pos` to `end_pos` in the `in_array`
# by space. If the joined text matches any value in multiple_words_icons, replace
# the element at the `start_pos` with the joined text. All elements after the `end_pos`
# which be moved forward to be at the `start_pos+1`
# Example:
# Running process_multiple_words_icon(["a", "ABS", "TIME", "b", "c"], 1, 2, ["ABS TIME"])
# outputs ["a", "ABS TIME", "b", "c"]
def process_multiple_words_icon(in_array, start_pos, end_pos, multiple_words_icons):
    joined_text = " ".join(in_array[start_pos:end_pos+1])
    if joined_text in multiple_words_icons:
        return in_array[0:start_pos] + [joined_text] + in_array[end_pos+1:]
    else:
        return in_array


# The first a few columns have texts in upper case. The next
# one or more columns have texts in lower case. These lower
# case texts should be joined by space to compose the target
# word meaning of the encoding of Bliss symbos in the first
# a few columns.
def process_raw_encoding(line, multiple_words_icons):
    columns_in = line.split()
    columns_out = []
    for index, text in enumerate(columns_in):
        if text.isupper():
            columns_out.append(text)
        elif text in ["*", "|"]:  # skip special characters
            continue
        else:
            columns_out.append(' '.join(columns_in[index:]))
            break

    # process icons composed by multiple words
    columns_out = process_multiple_words_icon(columns_out, 0, 1, multiple_words_icons)
    columns_out = process_multiple_words_icon(columns_out, 1, 2, multiple_words_icons)
    columns_out = process_multiple_words_icon(columns_out, 2, 3, multiple_words_icons)
    return columns_out


# Find the Bliss id for the given text:
# 1. Search through the given map first. If found, return id. Otherwise, continue the next step;
# 2. Search through the explanation JSON file. If found, return id. Otherwise, return None.
def get_bliss_id(text, explanation_json_data):
    text = text.lower()

    for item in explanation_json_data:
        defs = item["description"].lower().split(',')

        # Skip old descriptions
        if defs[-1].endswith("_(OLD)"):
            continue
        # The suffix "-(to)" indicates a verb. It's not part of the definition
        if defs[-1].endswith("-(to)"):
            defs[-1] = defs[-1][:-6]

        if text in defs:
            return item["id"]

    return None


# Provide the directory path as a parameter when running the script
source_bmw_path = sys.argv[1]
source_txt_path = sys.argv[2]
bliss_explanation_json_location = sys.argv[3]
output_json_location = sys.argv[4]
output_error_location = sys.argv[5]

missing_encodings = {}
error_rows = []

# load bliss translation json file
with open(bliss_explanation_json_location, 'r') as file:
    bliss_explanation_json = json.load(file)

# load bmw.json
with open(source_bmw_path, 'r') as file:
    bmw_json = json.load(file)
    existing_encodings = bmw_json["encodings"]

missing_bliss_id_texts = set()

multiple_words_icons = ["NOUN PL.", "ABS TIME", "N PERSON", "N.PER PL"]

# Iterate over all text files in the directory
for filename in os.listdir(source_txt_path):
    if filename.endswith('.txt'):
        txt_path = os.path.join(source_txt_path, filename)

        with open(txt_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    # skip empty lines
                    continue

                columns = process_raw_encoding(line, multiple_words_icons)
                if len(columns) < 2 or len(columns) > 4:
                    print(f"Error: line {line_number} in {filename} with content '{line}' - Row should have 2 to 4 columns.")
                    error_rows.append(line)
                    continue

                one_encoding = {}
                encoding_ids = []
                message = ""
                has_error = False
                message_found = False
                for text in columns:
                    if text.isupper():
                        # the encoding of bliss symbol texts should be in upper case
                        encoding_ids.append(text)
                        bliss_id = get_bliss_id(text, bliss_explanation_json)
                        if bliss_id is None:
                            # report the error when a BCI-AV-ID is not found
                            print(f"Error: line {line_number} in {filename} with content '{line}' - The Bliss id for '{text}' is not found.")
                            has_error = True
                            missing_bliss_id_texts.add(text)
                    else:
                        # the lower case text is the target message delivered by the Bliss symbol encoding
                        message = text
                        message_found = True

                if not message_found:
                    # error occurs when a line only has upper case letters. The expected format should have the target message
                    # in lower case. When it's not found, report an error.
                    print(f"Error: line {line_number} in {filename} with content '{line}' - The target message is not found in this line.")
                    has_error = True

                if has_error:
                    error_rows.append(line)
                else:
                    if message in existing_encodings and existing_encodings[message]["encoding"] != encoding_ids or message not in existing_encodings:
                        missing_encodings[message] = {}
                        missing_encodings[message]["encoding"] = encoding_ids
                        bliss_id = get_bliss_id(message, bliss_explanation_json)
                        missing_encodings[message]["bci-av-id"] = int(bliss_id) if isinstance(bliss_id, str) else bliss_id

# Write the JSON into a file
with open(output_json_location, "w") as json_file:
    json_file.write(json.dumps(missing_encodings, indent=4))
print(f"The final JSON is written into {output_json_location}")

# Write rows with errors into the given error file
if len(error_rows) > 0:
    with open(output_error_location, "w") as error_file:
        for error_row in error_rows:
            error_file.write(error_row + "\n")
    print(f"The error rows are written into {output_error_location}")
else:
    print("No error detected.")

# Report icons whose corresponding Bliss IDs are not found
if len(missing_bliss_id_texts) > 0:
    print(f"Missing Bliss IDs for these texts: {missing_bliss_id_texts}")
