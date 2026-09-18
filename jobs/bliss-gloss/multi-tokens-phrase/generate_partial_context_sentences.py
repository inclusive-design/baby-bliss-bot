# python generate_partial_context_sentences.py "./data/dataset_24918_lowness_shortness.py" '[" lowness", " shortness"]'

import sys
import json


def process_sentences(sentences, phrases):
    result = []
    for sentence in sentences:
        all_indices = []
        for phrase in phrases:
            if not phrase:
                continue
            start = 0
            while True:
                index = sentence.find(phrase, start)
                if index == -1:
                    break
                all_indices.append(index)
                start = index + 1
        if not all_indices:
            result.append(sentence)
            # print(f"Error: No phrase found in sentence: '{sentence}'")
            continue
        last_phrase_start = max(all_indices)
        truncated = sentence[:last_phrase_start]
        cleaned = truncated.rstrip()
        result.append(cleaned)
    return result


if len(sys.argv) != 3:
    print("Usage: python generate_partial_context_sentences.py <filename> <phrases>")
    print("Example: python generate_partial_context_sentences.py \"./data/data_24918.py\" '[\"phrase1\", \"phrase2\"]'")
    exit()

filename = sys.argv[1]
phrases = json.loads(sys.argv[2])
target_variable_names = ["training_positive_context_sentences", "training_negative_context_sentences"]

# Read and execute the file to extract the variable
namespace = {}
try:
    with open(filename, "r") as f:
        file_content = f.read()
    exec(file_content, namespace)
except Exception as e:
    print(f"Error reading or executing the file: {e}")
    exit()

# Retrieve the target variables
sentences = []
for target_variable in target_variable_names:
    if target_variable in namespace:
        sentences.extend(namespace[target_variable])

        # Process the sentences
        processed = process_sentences(sentences, phrases)

        with open(filename, "a") as f:
            f.write(f"\n{target_variable} = {json.dumps(processed, indent=4, ensure_ascii=False)}\n")
        print(f"Processed list appended to {filename} as '{target_variable}'.")

    else:
        print(f"Error: '{target_variable}' not found in the file.")
        exit()
