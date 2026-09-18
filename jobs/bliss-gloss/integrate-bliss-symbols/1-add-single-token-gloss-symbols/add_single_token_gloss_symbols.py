# Usage:
# python ~/bliss_gloss/add_single_token_gloss_symbols.py ./data/bliss_gloss_cleaned.json ./outputs/bliss_ids_added.json ./outputs/bliss_ids_not_added.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if len(sys.argv) != 4:
    print("Usage: python add_single_token_gloss_symbols.py <input_gloss_json> <output_file_with_added_id> <output_file_with_not_added_id>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
output_file_with_added_id = sys.argv[2]
output_file_with_not_added_id = sys.argv[3]

# Load the JSON file
with open(input_gloss_json, 'r') as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
# Build on top of the Llama-3.1-8B-Instruct model that is already fine-tuned on the symbol ID 24918
# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
output_model_path = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/integrate_single_token_gloss_symbols"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)


# Loop through the given glosses and find symbols that satisfy the following conditions:
# 1. The symbol only has one gloss
# 2. This single gloss is a single token in the tokenizer
# The function then returns the information of this symbol:
# a tuple of (gloss, input_embedding_of_gloss, output_embedding_of_gloss)
# If the symbol does not satisfy the above conditions, return (None, None, None)
def get_single_token_embeddings(model, tokenizer, glosses):
    if len(glosses) == 1:
        # Preprocess the gloss to remove spaces and dashes
        # and add a space at the beginning to match the tokenizer's encoding
        gloss = f" {glosses[0].replace(' ', '').replace('-', '')}"
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_id) == 1:
            return gloss, model.get_input_embeddings().weight[token_id], model.lm_head.weight[token_id]

    return None, None, None


not_added_bliss_ids = {}
added_bliss_ids = {}
new_tokens_to_add = []
new_token_input_embeddings = []
new_token_output_embeddings = []

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    gloss, input_emb, output_emb = get_single_token_embeddings(model, tokenizer, glosses)

    if gloss:
        new_token = f"[BLISS_{bliss_id}]"
        new_tokens_to_add.append(new_token)
        new_token_input_embeddings.append(input_emb)
        new_token_output_embeddings.append(output_emb)
        added_bliss_ids[bliss_id] = gloss
    else:
        not_added_bliss_ids[bliss_id] = glosses

# Add all new tokens at once
num_added_bliss_ids = tokenizer.add_tokens(new_tokens_to_add)

# Resize token embeddings once
model.resize_token_embeddings(len(tokenizer))

# Update embeddings for new tokens
with torch.no_grad():
    # Add new Bliss tokens
    for i, new_token in enumerate(new_tokens_to_add):
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        model.get_input_embeddings().weight[new_token_id] = new_token_input_embeddings[i]
        model.lm_head.weight[new_token_id] = new_token_output_embeddings[i]

# Save the updated model
print("Saving updated model...")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"Model saved to {output_model_path}")

print(f"{len(added_bliss_ids)} are added.")
print(f"{len(not_added_bliss_ids)} are not added.")

# Save added tokens to file
with open(output_file_with_added_id, 'w') as f:
    json.dump(added_bliss_ids, f, indent=2)
    print(f"Added tokens saved to {output_file_with_added_id}")

# Save not added tokens to file
with open(output_file_with_not_added_id, 'w') as f:
    json.dump(not_added_bliss_ids, f, indent=2)
    print(f"Not added tokens saved to {output_file_with_not_added_id}")
