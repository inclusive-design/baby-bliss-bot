# python compare_cosine_similarity.py <output_json_file>
# python compare_cosine_similarity.py ./test_results/compare_input_embeddings_cosine_similarity_result.json

# This script compares the cosine similarity between different glosses for given IDs on their input embeddings.

import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from itertools import combinations

if len(sys.argv) != 2:
    print("Usage: python compare_cosine_similarity.py <output_json_file>")
    sys.exit(1)

output_file = sys.argv[1]

input_glosses = {
    "24852": [" break", " fracture", " injury", " damage"],
    "23085": [" floor", " level"],
    "23409": [" attachment", " joint", " seam", " appendix", " annex"],
    "24887": [" dispersion", " dissemination", " scattering", " spread", " spreading"]
}
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
model.to(device)

results = {}

# Iterate through each ID in the data
for id, glosses in input_glosses.items():
    print(f"Processing ID: {id}")
    input_embeddings = []
    # Get input embeddings for all glosses of the current ID
    for gloss in glosses:
        tokens = tokenizer.tokenize(gloss)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"Gloss: {gloss}, Token IDs: {token_ids}")
        input_embeddings.append(model.get_input_embeddings().weight.data[token_ids[0]])

    # Calculate cosine similarity for each pair of glosses
    similarity_results = {}
    for (i, j) in combinations(range(len(glosses)), 2):
        similarity = torch.nn.functional.cosine_similarity(input_embeddings[i], input_embeddings[j], dim=0).item()
        pair_key = f"{glosses[i]} vs {glosses[j]}"
        similarity_results[pair_key] = similarity

    results[id] = similarity_results

# Write the results to the output file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results have been saved to {output_file}")
