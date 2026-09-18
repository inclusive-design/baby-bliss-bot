# Usage: python input_embedding_optimization.py 0.01 500

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import dataset_29111_wool_shop as dataset
from utils import create_training_data, calc_embeddings, evaluate_new_token  # noqa: E402
# from data import dataset_29111_wool_shop as dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import create_training_data, calc_embeddings, evaluate_new_token  # noqa: E402


def get_kv_cache_for_context(model, tokenizer, context):
    """Return the kv cache of the given context sentence and the current sequence length"""
    # Add the phrase to the context
    inputs = tokenizer(context, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Process the full sequence and return the updated KV cache
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True
        )

    return outputs.past_key_values, inputs.input_ids.shape[1]


def get_target_contextual_embedding(model, tokenizer, phrase, kv_cache_info):
    """Get contextual embedding of the last token in the phrase, using KV cache if available"""
    past_key_values, past_seq_len = kv_cache_info

    # Only tokenize the phrase
    phrase_inputs = tokenizer(phrase, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        # Use the KV cache for the context, only process the new tokens
        outputs = model(
            input_ids=phrase_inputs.input_ids,
            attention_mask=torch.ones(1, past_seq_len + phrase_inputs.input_ids.shape[1], device=model.device),
            past_key_values=past_key_values,
            output_hidden_states=True
        )

    return outputs.hidden_states[-1][0, -1, :]


def get_contextual_embedding(model, context, input_embedding):
    """Get contextual embedding using KV cache if available"""
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Process the full sequence and return the updated KV cache
        context_outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True
        )

    # Create inputs_embeds from the input_embedding (shape: [1, 1, embed_dim])
    inputs_embeds = input_embedding.unsqueeze(0).unsqueeze(0)

    outputs = model(
        # input_ids=new_token_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(1, inputs.input_ids.shape[1] + 1, device=model.device),
        past_key_values=DynamicCache.from_legacy_cache(context_outputs.past_key_values),   # Convert tuple-based cache to DynamicCache
        output_hidden_states=True,
        use_cache=False
    )

    return outputs.hidden_states[-1][0, -1, :]


def optimize_input_embedding(model, tokenizer, contexts, embed_dim, phrase, learning_rate, epochs, output_model_dir,
                             savepoint_ecpochs, checkpoint_epochs):
    """Optimize input embedding using KV caching for efficiency"""
    # Pre-compute target contextual embeddings and cache KV for all contexts
    target_contextuals = {}
    context_kv_cache = {}

    with torch.no_grad():
        for context in contexts:
            context_kv_cache[context] = get_kv_cache_for_context(model, tokenizer, context)

            # Get target embedding and cache KV states
            target_contextual = get_target_contextual_embedding(
                model, tokenizer, phrase, context_kv_cache[context]
            )
            target_contextuals[context] = target_contextual

    # Initialize input embedding and optimizer
    input_emb = torch.nn.Parameter(torch.randn(embed_dim, requires_grad=True, device=model.device))
    optimizer = torch.optim.AdamW([input_emb], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        all_losses = []

        # Calculate loss for each context
        for context in contexts:
            predicted_contextual = get_contextual_embedding(model, context, input_emb)

            # Loss based on contextual embedding similarity
            context_loss = 1 - F.cosine_similarity(
                predicted_contextual,
                target_contextuals[context],
                dim=0
            )

            all_losses.append(context_loss)

            # Keep track of the total loss value for printing
            total_loss += context_loss.item()

        # Sum all losses in one operation to create a clean computational graph
        if all_losses:
            accumulated_loss = torch.sum(torch.stack(all_losses))

            # Backpropagate with the accumulated loss
            accumulated_loss.backward(retain_graph=True)

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_([input_emb], max_norm=1.0)

            # Step the optimizer after accumulating all gradients
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

        if output_model_dir and (epoch + 1) >= savepoint_ecpochs and (epoch + 1) % checkpoint_epochs == 0:
            with torch.no_grad():
                # Store the current embedding
                model.get_input_embeddings().weight[new_token_id] = input_emb.clone()
            current_save_dir = f"{output_model_dir}_lr{learning_rate}/epochs{epoch + 1}"
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)

            tokenizer.save_pretrained(current_save_dir)
            model.save_pretrained(current_save_dir)

    if (epoch + 1) % 100 != 0:
        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

    return input_emb


if len(sys.argv) != 3:
    print("Usage: python input_embedding_optimization.py <learning_rate> <epochs>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])

# Initial values
# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
output_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding"
# output_model_dir = os.path.expanduser("~") + "/bliss_gloss/multi-tokens-phrase/test_results/models/optimize_input_embedding"
phrase = "wool shop"
target_tokens = [" wool", " yarn"]
new_token = "[BLISS_29111]"

savepoint_ecpochs = 700   # Start to save model after this number of epochs
checkpoint_epochs = 100   # Save model every this number of epoch

training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Track the total running time of this script
start_time = time.time()

# Calculate output embedding for the new token
# Get token ids for the target tokens
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

hidden_states, target_logits = create_training_data(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
)
new_output_embedding = calc_embeddings(hidden_states, target_logits)
# new_output_embedding = optimize_embeddings(model, hidden_states, target_logits, model.config.hidden_size, epochs, learning_rate)

end_time_calc_output_embedding = time.time()
elapsed_time = end_time_calc_output_embedding - start_time
print(f"Execution time for calculating output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Add the new token to the model with the new output embedding. The optimization of the input embedding needs it.
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

with torch.no_grad():
    model.get_output_embeddings().weight[new_token_id] = new_output_embedding
    # model.get_input_embeddings().weight[new_token_id] = torch.randn(model.config.hidden_size, requires_grad=True, device=model.device)

# Optimization of the input embedding
new_input_embedding = optimize_input_embedding(
    model,
    tokenizer,
    training_positive_context_sentences + training_negative_context_sentences,
    model.config.hidden_size,
    phrase,
    learning_rate,
    epochs,
    output_model_dir,
    savepoint_ecpochs,
    checkpoint_epochs
)

with torch.no_grad():
    model.get_input_embeddings().weight[new_token_id] = new_input_embedding

end_time_calc_input_embedding = time.time()
elapsed_time = end_time_calc_input_embedding - end_time_calc_output_embedding
print(f"Execution time for calculating input embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Save the last model if it is not saved yet
if output_model_dir and epochs % checkpoint_epochs != 0:
    current_save_dir = f"{output_model_dir}_lr{learning_rate}/epochs{epochs}"
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)
        tokenizer.save_pretrained(current_save_dir)
        model.save_pretrained(current_save_dir)

end_time_save_model = time.time()
elapsed_time = end_time_save_model - end_time_calc_input_embedding
print(f"Execution time for calculating input embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Validate the new token
# Test 1: Embedding initialization
if torch.all(model.get_input_embeddings().weight[new_token_id] == 0):
    print("Error: Input embedding not initialized!")
if torch.all(model.lm_head.weight[new_token_id] == 0):
    print("Error: Output embedding not initialized!")

# # Test 2: Test token prediction
# results = test_token_prediction(model, tokenizer, training_positive_context_sentences, new_token_id, target_token_ids)
# print_results("Re-verify predictions on POSITIVE training sentences:", target_tokens, new_token, results)
# results = test_token_prediction(model, tokenizer, training_negative_context_sentences, new_token_id, target_token_ids)
# print_results("Re-verify predictions on NEGATIVE training sentences:", target_tokens, new_token, results)
# results = test_token_prediction(model, tokenizer, testing_context_sentences, new_token_id, target_token_ids)
# print_results("Predictions on TESTING training sentences:", target_tokens, new_token, results)

# # Test 3: Generation
# print("\nValidation - Generation:")

# prompts = [
#     f"The {phrase} sells a variety of products including",
#     f"I stopped by the {phrase} to",
#     f"A cozy {phrase} is",
#     f"After visiting the {phrase}, I",
#     f"I met a friendly alpaca farmer at the {phrase} who",
#     f"I asked the owner of the {phrase} for",
#     f"The {phrase} had",
#     f"She spent hours at the {phrase}"
# ]
# for prompt in prompts:
#     print(f"Prompt: {prompt}")
#     print(f"Generated text with {phrase}: {generate_text(model, tokenizer, prompt)}")
#     print(f"Generated text with {new_token}: {generate_text(model, tokenizer, prompt.replace(phrase, new_token))}\n")

evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_context_sentences, new_token_id, target_token_ids, target_tokens, new_token, f" {phrase}"
)

end_time_validation = time.time()
elapsed_time = end_time_validation - end_time_calc_input_embedding
print(f"Execution time for calculating input embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
