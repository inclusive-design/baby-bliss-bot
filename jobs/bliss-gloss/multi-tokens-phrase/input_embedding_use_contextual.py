# Usage: python input_embedding_use_contextual.py

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import dataset_29111_wool_shop as dataset
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
from utils import create_training_data, calc_embeddings, add_token_to_model, test_token_prediction  # noqa: E402


def get_contextual_embedding(model, tokenizer, phrase):
    # Initialize input embedding using contextual embedding
    inputs = tokenizer(phrase, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use last token's hidden state from the final layer
    return outputs.hidden_states[-1][0, -1, :]


def print_results(title, target_tokens, new_token, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_tokens}: {result['rank_of_target_token_ids']}")
        print(f"Rank of {new_token}: {result['rank_of_new_token_id'][0]}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Initial values
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
phrase = "wool shop"
target_tokens = [" wool", " yarn", " shop"]
new_token = "[BLISS_29111]"

training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Get contextual embedding for the new token
new_input_embedding = get_contextual_embedding(model, tokenizer, phrase)

# Calculate output embedding for the new token
# Get token ids for the target tokens
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

hidden_states, target_logits = create_training_data(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
)
new_output_embedding = calc_embeddings(hidden_states, target_logits)

# Calculate cosine similarity between input and output embeddings
cos_sim = F.cosine_similarity(new_input_embedding.unsqueeze(0), new_output_embedding.unsqueeze(0))
print(f"\nCosine similarity between input and output embeddings: {cos_sim.item():.4f}")

# Add new token and resize embeddings
new_token_id = add_token_to_model(model, tokenizer, new_token, new_input_embedding, new_output_embedding)

# Validate the new token
# Test 1: Embedding initialization
if torch.all(model.get_input_embeddings().weight[new_token_id] == 0):
    print("Error: Input embedding not initialized!")
if torch.all(model.lm_head.weight[new_token_id] == 0):
    print("Error: Output embedding not initialized!")

# Test 2: Test token prediction
results = test_token_prediction(model, tokenizer, training_positive_context_sentences, new_token_id, target_token_ids)
print_results("Re-verify predictions on POSITIVE training sentences:", target_tokens, new_token, results)
results = test_token_prediction(model, tokenizer, training_negative_context_sentences, new_token_id, target_token_ids)
print_results("Re-verify predictions on NEGATIVE training sentences:", target_tokens, new_token, results)
results = test_token_prediction(model, tokenizer, testing_context_sentences, new_token_id, target_token_ids)
print_results("Predictions on TESTING training sentences:", target_tokens, new_token, results)

# Test 3: Generation
print("\nValidation - Generation:")

prompts = [
    f"The {phrase} sells a variety of products including",
    f"I stopped by the {phrase} to",
    f"A cozy {phrase} is",
    f"After visiting the {phrase}, I",
    f"I met a friendly alpaca farmer at the {phrase} who",
    f"I asked the owner of the {phrase} for",
    f"The {phrase} had",
    f"She spent hours at the {phrase}"
]
for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Generated text with {phrase}: {generate_text(model, tokenizer, prompt)}")
    print(f"Generated text with {new_token}: {generate_text(model, tokenizer, prompt.replace(phrase, new_token))}\n")
