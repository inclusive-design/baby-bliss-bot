# Usage: python input_embedding_optimization_new_model_token.py

# This script implements an optimization method for teaching a language model the meaning of a
# multi-token phrase by optimizing the input embedding of a single, newly added token. The goal
# is to have this new token effectively capture the semantics of the original phrase without modifying
# the LLM's existing weights. This approach requires further fine-tuning of the model to ensure that the
# new token can be used in various contexts.

# The core of the method is to add a special token to the model's vocabulary and then optimize only its
# input embedding. The optimization objective is to align the contextual representation of this new token
# with that of the original phrase by minimizing the cosine distance between their respective final hidden
# states when appended to a set of prefix sentences.

# The process involves these steps:
# 1. Data Preparation: Create a dataset of training sentences, validation sentences, and testing prompts.
# 2. Initialization: The new token is initialized with an average embedding from the phrase's constituent
# tokens and a pre-calculated output embedding to have a good starting point.
# 3. Isolated Optimization: With all other parameters in the base model frozen except the input embedding
# layer, the script performs iterative optimization. A gradient mask is applied during the backward pass
# to isolate updates exclusively to the new token's input embedding.
# 4. Selection & Evaluation: The optimal embedding is selected based on the lowest validation loss. The
# final model is then evaluated with text generation and next-word prediction tasks to verify that the new
# token has effectively captured the semantics of the original phrase.

# The test result is located at
# `test_results/29111_wool_shop/input_embedding_optimization_new_model_token_0.0005_100.log`
# 1. The results show strong convergence, with the validation loss decreasing from an initial value of 0.083170
# to a final best of 0.020297 at epoch 86.
# 2. A comparison of the initial and final embeddings shows a cosine similarity of 0.3186 and an Euclidean distance
# of 1.0985, indicating significant refinement through optimization.
# 3. The new token demonstrates effective learning, achieving high accuracy in next-word prediction and text
# generation tasks.

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import create_training_data, calc_output_embedding, get_average_input_embeddings, evaluate_new_token  # noqa: E402


def get_target_contextual_embedding(model, tokenizer, prefix_sentences, phrase):
    """Get contextual embedding of the last token in the phrase when the phrase is appended to each of the prefix sentences"""
    target_hidden_states = {}
    with torch.no_grad():
        for prefix in prefix_sentences:
            text = prefix + phrase
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)

            hidden_state_reference = outputs.hidden_states[-1][:, -1, :].detach()
            target_hidden_states[prefix] = hidden_state_reference

    return target_hidden_states


def calculate_loss(model, tokenizer, prefix_sentence, new_token, target_hidden_state, loss_fn, device):
    sentence_with_new_token = prefix_sentence + new_token
    sentence_with_new_token_inputs = tokenizer(sentence_with_new_token, return_tensors="pt").to(device)
    sentence_with_new_token_outputs = model(**sentence_with_new_token_inputs, output_hidden_states=True)
    current_hidden_state = sentence_with_new_token_outputs.hidden_states[-1][:, -1, :]

    # --- Calculate Loss ---
    target = torch.ones(current_hidden_state.shape[0]).to(device)
    loss = loss_fn(current_hidden_state, target_hidden_state, target)
    return loss


# Main script
LEARNING_RATE = 5e-4
EPOCHS = 100
savepoint_epochs = 90   # Start to save model after this number of epochs
checkpoint_epochs = 5   # Save model every this number of epoch

initial_dataset_file = os.path.expanduser("~") + "/bliss_gloss/multi-tokens-phrase/data/dataset_29111_wool_shop.py"

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
initial_output_embedding = calc_output_embedding(hidden_states, target_logits)

average_input_embedding = get_average_input_embeddings(model, tokenizer, [phrase])

# Add the new token to the model with the new output embedding. The optimization of the input embedding needs it.
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

with torch.no_grad():
    model.get_output_embeddings().weight[new_token_id] = initial_output_embedding.clone()
    model.get_input_embeddings().weight[new_token_id] = average_input_embedding.clone()

end_time_calc_embedding = time.time()
elapsed_time = end_time_calc_embedding - start_time
print(f"Execution time for calculating and setting initial input and output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Get target contextual embedding for the optimization
print("Calculating target hidden states of target phrases and validation phrases before the optimization...")
model.eval()
target_hidden_states = get_target_contextual_embedding(model, tokenizer, training_positive_context_sentences, phrase)
validation_hidden_states = get_target_contextual_embedding(model, tokenizer, validation_positive_context_sentences, phrase)

end_time_get_target_hidden_states = time.time()
elapsed_time = end_time_get_target_hidden_states - end_time_calc_embedding
print(f"Execution time for calculating hidden states: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Freeze Model and Unfreeze Only the New Input Embedding
model.train()
print("Freezing all model parameters...")
for param in model.parameters():
    param.requires_grad = False

print(f"Unfreezing the input embedding for the new token '{new_token}'...")
input_embeddings = model.get_input_embeddings()
input_embeddings.weight.requires_grad = True

# Optimization
print("Starting optimization of the input embedding...")
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
loss_fn = torch.nn.CosineEmbeddingLoss()

# Track the best validation loss and the best input embedding
best_validation_loss = float('inf')
best_input_embedding = None

for epoch in range(EPOCHS):
    # Set the model to training mode. The model will be switched to eval mode during validation.
    model.train()
    total_loss = 0
    # Iterate through the prefixes and their pre-calculated target hidden states
    for prefix_sentence, target_hidden_state in target_hidden_states.items():
        optimizer.zero_grad()

        loss = calculate_loss(model, tokenizer, prefix_sentence, new_token, target_hidden_state, loss_fn, device)

        # Backpropagate and Update
        loss.backward()

        # Create a mask that keeps only the gradient for new_token_id
        mask = torch.zeros_like(input_embeddings.weight.grad)
        mask[new_token_id, :] = 1  # Set the row for new_token_id to 1

        # Applying the mask to zero out gradients for all other embeddings
        input_embeddings.weight.grad *= mask

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(target_hidden_states)

    # Validation after every epoch
    model.eval()
    total_validation_loss = 0

    # Wrap the validation loop with torch.no_grad()
    with torch.no_grad():
        for prefix_sentence, target_hidden_state in validation_hidden_states.items():
            validation_loss = calculate_loss(model, tokenizer, prefix_sentence, new_token, target_hidden_state, loss_fn, device)
            total_validation_loss += validation_loss.item()

    avg_validation_loss = total_validation_loss / len(validation_hidden_states)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Average Training Loss: {avg_train_loss:.6f}, Average Validation Loss: {avg_validation_loss:.6f}")

    # Save the best input embedding if the validation loss has improved
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        best_input_embedding = model.get_input_embeddings().weight[new_token_id].detach().clone()
        print(f"**  New best validation loss: {best_validation_loss:.6f}. Saved the best input embedding.")

print(f"\nOptimization finished! Best validation loss: {best_validation_loss:.6f}")

end_time_optimization = time.time()
elapsed_time = end_time_optimization - end_time_get_target_hidden_states
print(f"Execution time for the optimization: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

if best_input_embedding is not None and output_model_dir:
    print(f"Assigning the best embedding (with the validation loss {best_validation_loss:.6f}) back to the model.")

    with torch.no_grad():
        model.get_input_embeddings().weight[new_token_id] = best_input_embedding

    if output_model_dir:
        best_model_dir = f"{output_model_dir}_lr{LEARNING_RATE}"
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        print(f"Saving the final best model to {best_model_dir}")
        tokenizer.save_pretrained(best_model_dir)
        model.save_pretrained(best_model_dir)
    else:
        print(f"No model saved. Dir not exist: {output_model_dir}")

input_embedding_similarity = torch.nn.functional.cosine_similarity(average_input_embedding.to(model.device), best_input_embedding.to(model.device), dim=0)
print(f"Cosine Similarity of the new token input embedding before and after: {input_embedding_similarity:.4f}")
distance = torch.norm(average_input_embedding.to(model.device) - best_input_embedding.to(model.device), p=2)
print(f"Euclidean Distance of the new token input embedding before and after: {distance.item():.4f}")

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
elapsed_time = end_time_validation - end_time_optimization
print(f"Execution time for evaluation: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

total_time = end_time_validation - start_time
print(f"\nTotal execution time: {int(total_time // 60)} minutes and {total_time % 60:.2f} seconds")
