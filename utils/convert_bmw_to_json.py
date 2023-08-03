'''
This script reads txt files that contain BMW encoding list from a directory and convert them into JSON.
The script will report error when BCI-AV-ID for a text is not found or the target message is not found.
At the end of execution, a set of texts that don't have the matching BCI-AV-ID will be summarized and
reported.

Errors will be written into the given error file. The converted JSON will be written into the given JSON file.

These errors are reported:
1. The length of icons in an encoding is less than 2 or longer than 4.
2. The message conveyed by the encoding is not found. This is because the message should be in lower case
   in the scanned document. When all texts are in upper case, this error is reported.

Usage: python convert_bmw_to_json.py source_txt_path bliss_translation_json_location output_json_location output_error_location
Parameters:
  source_txt_path: The path where text files are
  bliss_translation_json_location: The location of the JSON file that contains the translation between Bliss
  BCI-AV-ID and its language translation
  output_json_location: The location of the output JSON file. If it doesn't exist, the script will create it
  output_error_location: The location of the error file that rows in input files not processed due to any error are written into
Return: None

Example: python convert_bmw_to_json.py ~/Downloads/bmw_texts/ ../data/bliss_symbol_explanations.json ../data/bmw.json ../data/error.txt

File formats:
1. The content of `source_txt_path`
```
SAY THINK VERB+S talks
SAY TO+VERB to say
SAY VERB say
```
Using `SAY THINK VERB+S talks` as an example, "SAY", "THINK" and "VERB+S" are three keys pressed on the keyboard. It
delivers a message of "talks".

2. The generated JSON structure for BMW encoding is:
```
[
    "encodings": {
        "talked": {
            "encoding": [
                "SAY",
                "THINK",
                "VERB+ED"
            ],
            "bci-av-id": 123
        }
    ...
    },
     "word_to_id_map": {
        "VERB": 12335,
        "VERB+S": [12335, "/", 8499],
        ...
    }
]
```
Note values for BCI-AV-ID information, such as `encodings.talked.bci-av-id` and `word_to_id_map.VERB`
can be an array if this Bliss symbol is composed by multiple symbols. The use of "/" or ";" means:
* In the format of [12335, "/", 8499], the Bliss character 12335 and the Bliss character of 8499 are
displayed side by side;
* In the format of [12335, ";", 8499], the Bliss character 12335 and the indicator 8499 are displayed
in the way that the indicator 8499 is on top of the the charactor 12355;
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


# # Search through bliss description JSON file to find the Bliss id for the input string
# def find_id_in_explanation_json(input_string, json_data):
#     input_string = input_string.lower()

#     for item in json_data:
#         descriptions = item["description"].lower().split(',')
#         if input_string in descriptions:
#             return item["id"]

#     return None  # Return None if the string is not found in any "description" field


# Find the Bliss id for the given text:
# 1. Search through the given map first. If found, return id. Otherwise, continue the next step;
# 2. Search through the explanation JSON file. If found, return id. Otherwise, return None.
def get_bliss_id(text, word_to_bci_av_id_map, explanation_json_data):
    text = text.lower()

    for key, value in word_to_bci_av_id_map.items():
        if text == key.lower():
            return value

    for item in explanation_json_data:
        descriptions = item["description"].lower().split(',')
        if text in descriptions:
            return item["id"]

    return None


# Provide the directory path as a parameter when running the script
source_txt_path = sys.argv[1]
bliss_translation_json_location = sys.argv[2]
output_json_location = sys.argv[3]
output_error_location = sys.argv[4]

final_json = {
    "encodings": {},
    "word_to_id_map": {}
}
error_rows = []

# load bliss translation json file
with open(bliss_translation_json_location, 'r') as file:
    bliss_translation_json = json.load(file)

missing_bliss_id_texts = set()

multiple_words_icons = ["NOUN PL.", "ABS TIME", "N PERSON", "N.PER PL"]

final_json = {
    "encodings": {},
    "word_to_id_map": {
        "ABS TIME": ["HC8N:0,8;S4:2,10", "/", 17732],
        "ADJ.": 25554,
        "ADJ.+ER": 24879,
        "ADJ.+EST": 24944,
        "AM/BE": 12639,
        "COMMUN.": 13390,
        "CONJ.": 23409,
        "DEM.": [17720, "/", 17697],
        "DIRECTN": 13682,
        "EXCLAIM": 14947,
        "FORCE": 22914,
        "FURNITUR": 14416,
        "HAVE": 14685,
        "HE+": 14687,
        "I+": 14916,
        "INTERJ.": [24961, "/", 15947],
        "IT+": 14960,
        "LANG.": 15172,
        "LEGSFEET": 15189,
        "LIMITIME": 15212,
        "MINSPEAK": [14647, "/", 15972, "/", 15172],
        "N PERSON": 16161,
        "N.PER PL": [16161, ";", 9011],
        "NOUN PL.": [17717, ";", 9011],
        "OUTWORL": 15411,
        "OUTWORLD": 15411,
        "PREP.": [12324, "/", 17717],
        "PREVERB": [13867, "/", 12335],
        "PREVRB+S": [13867, "/", 12335],
        "POSS.": 24925,
        "RECEIVE": 14435,
        "THEY+": 17714,
        "TO+VERB": 13860,
        "YOU+": 18465,
        "SHE+": 14688,
        "SEXUAL": 16933,
        "VERB": 12335,
        "VERB+S": [12335, "/", 8499],
        "VERB+ED": [12335, "/", 15975],
        "VERB+EN": [12335, "/", 13949],
        "VERB+ING": [12335, "/", 14390],
        "WE+": 18212,
        "WRITE": 18285,
        "YES": 18294
    }
}

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
                        bliss_id = get_bliss_id(text, final_json["word_to_id_map"], bliss_translation_json)
                        if bliss_id is None:
                            # report the error when a BCI-AV-ID is not found
                            print(f"Error: line {line_number} in {filename} with content '{line}' - The Bliss id for '{text}' is not found.")
                            has_error = True
                            missing_bliss_id_texts.add(text)
                        else:
                            final_json["word_to_id_map"][text] = int(bliss_id) if isinstance(bliss_id, str) else bliss_id
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
                    if message not in final_json["encodings"]:
                        final_json["encodings"][message] = {}
                    final_json["encodings"][message]["encoding"] = encoding_ids
                    bliss_id = get_bliss_id(message, final_json["word_to_id_map"], bliss_translation_json)
                    final_json["encodings"][message]["bci-av-id"] = int(bliss_id) if isinstance(bliss_id, str) else bliss_id

# Write the JSON into a file
with open(output_json_location, "w") as json_file:
    json_file.write(json.dumps(final_json, indent=4))
print(f"The final JSON is written into {output_json_location}")

# Write rows with errors into the given error file
if len(error_rows) > 0:
    with open(output_error_location, "w") as error_file:
        for error_row in error_rows:
            error_file.write(error_row + "\n")
    print(f"The error rows are written into {output_error_location}")
else:
    print("No error detected.")

if len(missing_bliss_id_texts) > 0:
    print(f"Missing Bliss IDs for these texts: {missing_bliss_id_texts}")
