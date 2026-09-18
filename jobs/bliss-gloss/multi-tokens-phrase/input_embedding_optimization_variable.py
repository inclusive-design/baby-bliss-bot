# Usage: python input_embedding_optimization_variable.py

# This script implements an optimization method to calculate a single input embedding that
# semantically represents a multi-token phrase. The method involves optimizing a standalone "virtual token"
# to replicate the contextual impact of the original phrase on a language model without
# altering the model's underlying weights.

# The optimization objective is to minimize the difference between two contextual embeddings:
# 1) the target hidden state produced by the language model after processing a prefix sentence followed
# by the original phrase
# 2) the hidden state produced after processing the same prefix followed by the learnable virtual token.

# The process involves three main steps:
# 1. Semantic Initialization: The virtual token is initialized with the average of the static embeddings
# of the tokens in the phrase, providing a good starting point.
# 2. Optimization: With the base model's parameters frozen, an Adam optimizer iteratively refines the virtual
# token's embedding to minimize the Cosine Embedding Loss between its output and the target hidden states.
# The best-performing embedding is selected based on the lowest validation loss.
# 3. Qualitative Evaluation: The final, optimized embedding is assigned to a new token in the model's vocabulary
# and is evaluated on its ability to perform next-word prediction and coherent text generation.

# The test result is located at `test_results/29111_wool_shop/input_embedding_optimization_variable_0.0005_100.log`
# 1. The results show strong convergence, with the validation loss decreasing from an initial value of 0.083170
# to a final best of 0.021390 at epoch 87.
# 2. A comparison of the initial and final embeddings shows a cosine similarity of 0.3186 and an Euclidean distance
# of 1.1064, indicating significant refinement through optimization.
# 3. The new token demonstrates effective learning, achieving high accuracy in next-word prediction and text
# generation tasks.

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import create_training_data, calc_output_embedding, optimize_input_embedding, get_average_input_embeddings, evaluate_new_token

LEARNING_RATE = 5e-4
EPOCHS = 100
savepoint_epochs = 90   # Start to save model after this number of epochs
checkpoint_epochs = 5   # Save model every this number of epoch

initial_dataset_file = os.path.expanduser("~") + "/bliss_gloss/multi-tokens-phrase/data/dataset_12914_to_brush_teeth.py"

if not os.path.exists(initial_dataset_file):
    print(f"Error: Initial dataset file '{initial_dataset_file}' does not exist.")
    sys.exit(1)

# Initial values
# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
output_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding"
# output_model_dir = os.path.expanduser("~") + "/bliss_gloss/multi-tokens-phrase/test_results/models/optimize_input_embedding"
phrase = "to brush teeth"
target_tokens = [" to"]
new_token = "[BLISS_12914]"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read and execute the file to extract the target variables
namespace = {}
target_variable_names_in_initial_dataset = ["training_positive_context_sentences", "validation_positive_context_sentences", "training_negative_context_sentences", "testing_positive_context_sentences", "testing_negative_context_sentences", "testing_text_generation_prompts"]
with open(initial_dataset_file, "r") as f:
    initial_dataset = f.read()
exec(initial_dataset, namespace)

training_positive_context_sentences = namespace["training_positive_context_sentences"]
validation_positive_context_sentences = namespace["validation_positive_context_sentences"]
training_negative_context_sentences = namespace["training_negative_context_sentences"]
testing_positive_context_sentences = namespace["testing_positive_context_sentences"]
testing_negative_context_sentences = namespace["testing_negative_context_sentences"]
testing_text_generation_prompts = namespace["testing_text_generation_prompts"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to(device)

# Track the total running time of this script
start_time = time.time()

print("Calculating initial input and output embeddings...")
# Calculate output embedding for the new token
# Get token ids for the target tokens
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

hidden_states, target_logits = create_training_data(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
)
calculated_output_embedding = calc_output_embedding(hidden_states, target_logits)

end_time_calc_output_embedding = time.time()
elapsed_time = end_time_calc_output_embedding - start_time
print(f"Execution time for calculating the output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

average_input_embedding = get_average_input_embeddings(model, tokenizer, [phrase])
optimized_input_embedding = optimize_input_embedding(
    model, tokenizer, EPOCHS, LEARNING_RATE, phrase,
    training_positive_context_sentences, validation_positive_context_sentences, device
)

end_time_calc_input_embedding = time.time()
elapsed_time = end_time_calc_input_embedding - end_time_calc_output_embedding
print(f"Execution time for calculating the input embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Add the new token to the model with the new output embedding. The optimization of the input embedding needs it.
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

with torch.no_grad():
    model.get_output_embeddings().weight[new_token_id] = calculated_output_embedding
    model.get_input_embeddings().weight[new_token_id] = optimized_input_embedding

input_embedding_similarity = torch.nn.functional.cosine_similarity(average_input_embedding.to(model.device), optimized_input_embedding.to(model.device), dim=0)
print(f"Cosine Similarity of the new token input embedding before and after: {input_embedding_similarity.item():.4f}")
distance = torch.norm(average_input_embedding.to(model.device) - optimized_input_embedding.to(model.device), p=2)
print(f"Euclidean Distance of the new token input embedding before and after: {distance.item():.4f}")

end_time_add_new_token = time.time()
elapsed_time = end_time_add_new_token - end_time_calc_output_embedding
print(f"Execution time for adding a new token: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Evaluation
# Test 1: Embedding initialization
if torch.all(model.get_input_embeddings().weight[new_token_id] == 0):
    print("Error: Input embedding not initialized!")
if torch.all(model.lm_head.weight[new_token_id] == 0):
    print("Error: Output embedding not initialized!")

evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, f" {phrase}"
)

end_time_validation = time.time()
elapsed_time = end_time_validation - end_time_add_new_token
print(f"Execution time for evaluation: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

total_time = end_time_validation - start_time
print(f"\nTotal execution time: {int(total_time // 60)} minutes and {total_time % 60:.2f} seconds")
