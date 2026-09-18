# Usage: python sentence_predictions.py '["lowness", "shortness"]'

"""
This script reports the rank of a certain token in the predictions in a given context.
"""

import os
import sys
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import dataset_24918_lowness_shortness as user_dataset
from utils import get_token_prediction, print_results
# from data import dataset_24918_lowness_shortness as user_dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import get_token_prediction, print_results  # noqa: E402

if len(sys.argv) != 2:
    print("Usage: python sentence_predictions.py <glosses>")
    sys.exit(1)

glosses = json.loads(sys.argv[1])

RANK_POSITIVE_THRESHOLD = 100
RANK_NEGATIVE_THRESHOLD = 1000

training_positive_context_sentences = user_dataset.training_positive_context_sentences
training_negative_context_sentences = user_dataset.training_negative_context_sentences
testing_positive_context_sentences = user_dataset.testing_positive_context_sentences
testing_negative_context_sentences = user_dataset.testing_negative_context_sentences

# Track the total running time of this script
start_time = time.time()

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Set the model to evaluation mode
model.eval()

# Get token ids for the target tokens
target_tokens = [tokenizer.tokenize(gloss)[0] for gloss in glosses]
target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)

print(f"Target tokens: {target_tokens}; Token IDs: {target_token_ids}")


def print_qualified(variable_name, target_tokens, results, verify_type="positive", threshold=RANK_POSITIVE_THRESHOLD):
    print(f"{variable_name} = [")
    for result in results:
        if (verify_type == "positive" and any(int(rank) < threshold for rank in result["rank_of_target_token_ids"])) or (verify_type == "negative" and all(int(rank) > threshold for rank in result["rank_of_target_token_ids"])):
            print(f'    "{result["context"]}",')
    print("]")


# Re-verify predictions on training sentences for the new token
results = get_token_prediction(model, tokenizer, training_positive_context_sentences, target_token_ids)
print_results("Results on POSITIVE training sentences:", target_tokens, None, results)
print_qualified("training_positive_context_sentences", target_tokens, results, "positive", RANK_POSITIVE_THRESHOLD)

results = get_token_prediction(model, tokenizer, training_negative_context_sentences, target_token_ids)
print_results("Results on NEGATIVE training sentences:", target_tokens, None, results)
print_qualified("training_negative_context_sentences", target_tokens, results, "negative", RANK_NEGATIVE_THRESHOLD)

# results = get_token_prediction(model, tokenizer, testing_positive_context_sentences, target_token_ids)
# # print_results("Results on TESTING POSITIVE testing sentences:", target_tokens, None, results)
# print_qualified("testing_positive_context_sentences", target_tokens, results, "positive", RANK_POSITIVE_THRESHOLD)

# results = get_token_prediction(model, tokenizer, testing_negative_context_sentences, target_token_ids)
# # print_results("Results on TESTING NEGATIVE testing sentences:", target_tokens, None, results)
# print_qualified("testing_negative_context_sentences:", target_tokens, results, "negative", RANK_NEGATIVE_THRESHOLD)

end_time = time.time()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - end_time
print(f"Execution time: {int(elapsed_time // 3600)} hours {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
