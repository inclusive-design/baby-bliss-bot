# This script reads in the Bliss gloss file. For each pair of glosses associated with a Bliss ID,
# it computes the cosine similarity between their input and output embeddings. If the similarity
# for either embedding is below a specified threshold, the pair is flagged as a mismatch. All
# mismatches are written into an output file.
#
# python compare_gloss_by_cosine.py data/bliss_gloss_cleaned_synonyms.json data/consine_mismatch.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utility_funcs import compare_embedding, preprocess_gloss, get_contextual_embedding

# Set a standard deviation threshold for detecting outliers
THRESHOLD = 0.8

if len(sys.argv) != 3:
    print("Usage: python compare_gloss_by_cosine.py <input_gloss_json> <analysis_output_file>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
analysis_output_file = sys.argv[2]

# Load the JSON file
with open(input_gloss_json, "r") as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

all_single_token_glosses = {}

bliss_ids_has_input_outliers = []
bliss_ids_has_output_outliers = []
count_mismatch = 0
count_total_comparison = 0

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    single_token_glosses = []
    contextual_embeddings = []

    # Loop through each gloss to find its token id and input/output embedding
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)

        # Check if it"s a single token gloss
        if len(token_id) != 1:
            continue

        token_id = token_id[0]  # Get the single token ID
        single_token_glosses.append(gloss)

        # Use the second contextual embedding because the first token is a special token "<|begin_of_text|>"
        contextual_embeddings.append(get_contextual_embedding(model, tokenizer, gloss)[0])

    num_of_single_glosses = len(single_token_glosses)
    if num_of_single_glosses > 1:
        all_single_token_glosses[bliss_id] = {
            "single_token_glosses": single_token_glosses,
            "mismatch": []
        }

        for i in range(num_of_single_glosses):
            for j in range(i + 1, num_of_single_glosses):
                count_total_comparison = count_total_comparison + 1
                contextual_comparison = compare_embedding(contextual_embeddings[i], contextual_embeddings[j])

                if contextual_comparison["similarity"] < THRESHOLD:
                    count_mismatch = count_mismatch + 1
                    one_mismatch = {
                        "glosses": [single_token_glosses[i], single_token_glosses[j]]
                    }
                    one_mismatch["contextual_similarity"] = contextual_comparison["similarity"]

                    all_single_token_glosses[bliss_id]["mismatch"].append(one_mismatch)

if len(all_single_token_glosses) > 0:
    print(f"{count_total_comparison} pairs compared. {count_mismatch} found with similarity lower than {THRESHOLD}")
    # Save analysis results to file
    with open(analysis_output_file, "w") as f:
        json.dump(all_single_token_glosses, f, indent=2)
