# python test_model.py <bliss_id> <glosses> <data_dir> <adapter_dir_name>
# python test_model.py 12356 '["air", "atmosphere"]' 'data' '19_adapter_12869'

# This script tests the fine-tuned model to ensure previously added new token stays intact
# when more new tokens are added.

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

# The path to the best model is saved in the Trainer state, and can be accessed using: trainer.state.best_model_checkpoint
# This path is also saved in "output_dir/checkpoint-<step>/trainer_state.json" -> "best_model_checkpoint" path

import os
import sys
import importlib
import time
import json
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from utils import test_text_generation, test_token_prediction, print_results

if len(sys.argv) != 5:
    print("Usage: python test_model.py <bliss_id> <glosses> <data_dir_name> <adapter_dir_name>")
    sys.exit(1)

# Track the total running time of this script
start_time = time.time()

bliss_id = sys.argv[1]
glosses = json.loads(sys.argv[2])
data_dir_name = sys.argv[3]
adapter_dir_name = sys.argv[4]

learning_rate = 0.0003
epochs = 10
batch_size = 10

# Dynamic import of the dataset
module_name = f"dataset_{bliss_id}_{'_'.join([re.sub(r'[^a-zA-Z0-9]', '', gloss) for gloss in glosses])}"

try:
    user_dataset = importlib.import_module(f"{data_dir_name}.{module_name}")
except ModuleNotFoundError:
    print(f"Error: Module '{module_name}' not found.")
    sys.exit(1)

root_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/"
base_model_dir = os.path.join(root_model_dir, "1_single_gloss_model")
new_token = f"[BLISS_{bliss_id}]"
dtype = torch.bfloat16

adapter_dir = os.path.join(root_model_dir, adapter_dir_name)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
print(f"Tokenizer loaded from the latest adapter directory: {adapter_dir}")

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Note: Must resize the model vocaubulary size before loading the adapter because the vocabulary
# size of the base model is less than the size of the tokenizer loaded from the adapter directory
# as new tokens were added to the adapter.
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, adapter_dir)
print(f"Loaded adapter weights from {adapter_dir}")
# End of loading the model and tokenizer

model.eval()

# Get the new token ID
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"New token '{new_token}' has ID: {new_token_id}.")

glosses_in_words = [" " + gloss for gloss in glosses]
print(f"Glosses in words: {glosses_in_words}")
target_tokens = [tokenizer.tokenize(token)[0] for token in glosses_in_words]
target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
print(f"Target tokens: {target_tokens}; Target token IDs: {target_token_ids}")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
testing_positive_context_sentences = user_dataset.testing_positive_context_sentences
testing_negative_context_sentences = user_dataset.testing_negative_context_sentences
testing_text_generation_prompts = user_dataset.testing_text_generation_prompts

results = test_token_prediction(model, tokenizer, testing_positive_context_sentences, target_token_ids, new_token_id)
print_results("Predictions on TESTING POSITIVE testing sentences:", target_tokens, new_token, results)
results = test_token_prediction(model, tokenizer, testing_negative_context_sentences, target_token_ids, new_token_id)
print_results("Predictions on TESTING NEGATIVE testing sentences:", target_tokens, new_token, results)

test_text_generation(
    model, tokenizer, testing_text_generation_prompts, new_token, glosses_in_words
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
