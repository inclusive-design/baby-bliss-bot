'''
Loops through bmw.json and reports all messages whose BCI-AV-ID is null.

Usage: python find_null_ids.py source_bmw_path
Parameters:
  source_bmw_path: The path where bmw.json is
Return: None

Example: python find_null_ids.py ../data/bmw.json
'''

import json
import sys

source_json_file = sys.argv[1]

with open(source_json_file, 'r') as file:
    data = json.load(file)
    for message, value in data["encodings"].items():
        if value["bci-av-id"] is None:
            print(message)
