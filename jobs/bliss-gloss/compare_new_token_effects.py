# This script is to test if the same input embedding will lead to the same output embedding
# after going through a language model.

# The test is to add two Bliss specific tokens to the Llama tokenizer by associating them
# with the embedding of word " light" and " bark" with space in front of the word. These
# two words are corresponding to one single token in Llama 3.1 8B model. Use a few sentences
# that have one of these two words, replace them with the new Bliss tokens, then compare
# the input and output embedding of sentences before and after the replacement.

# Usage: python compare_new_token_effects.py

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Load the LLaMA model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

word_to_token = {"[BLISS_1]": " light", "[BLISS_2]": " bark"}
special_tokens = list(word_to_token.keys())

# Add special tokens
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

# Set embeddings for special tokens
for special_token, word in word_to_token.items():
    tokens = tokenizer.tokenize(word)
    token_id = tokenizer.convert_tokens_to_ids(tokens)
    print(f"word: '{word}'; tokens: {tokens}; token_id: {token_id}")

    if len(token_id) > 1:
        print(f"Warning: '{word}' is tokenized into multiple tokens: {tokens}.")
        continue
    word_embedding = model.get_input_embeddings().weight[token_id].clone()
    special_token_index = tokenizer.convert_tokens_to_ids(special_token)
    print(f"special_token_index: {special_token_index}")
    model.get_input_embeddings().weight.data[special_token_index] = word_embedding
    print("===\n")


def replace_word_with_special_token(phrase, word, special_token, tokenizer):
    # Tokenize the original phrase
    original_tokens = tokenizer.tokenize(phrase)

    # Find the index of the word to replace
    word_index = original_tokens.index(f"▁{word}")

    # Replace the word with the special token
    original_tokens[word_index] = special_token

    # Join the tokens back into a string
    new_phrase = tokenizer.convert_tokens_to_string(original_tokens)

    return new_phrase


# Define phrases for comparison
phrases = {
    " light": [
        "She turned on the light in the room.",
        "She prefers to eat a light meal in the evening."
    ],
    " bark": [
        "The dog began to bark loudly at the stranger.",
        "The tree's bark was rough and covered in moss."
    ]
}


# Function to get embeddings for a phrase
def get_phrase_embedding(tokenizer, model, phrase):
    print(f"Phrase: {phrase}")
    tokens = tokenizer.tokenize(phrase)
    inputs = tokenizer(phrase, return_tensors="pt")
    print(f"tokens: {tokens}")
    print(f"token inputs: {inputs}\n")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    input_embedding = model.get_input_embeddings()(inputs['input_ids'])

    # Use the last hidden state from the output
    output_embedding = outputs.hidden_states[-1]

    return {
        "input_embedding": input_embedding,
        "output_embedding": output_embedding
    }


# Perform comparisons
for token, word in word_to_token.items():
    print(f"Comparing '{word}' with '{token}':")
    print("-" * 50)

    for phrase in phrases[word]:
        original_phrase = phrase
        special_phrase = phrase.replace(word, token)

        original_embedding = get_phrase_embedding(tokenizer, model, original_phrase)
        special_embedding = get_phrase_embedding(tokenizer, model, special_phrase)

        # Calculate cosine similarity for each token pair
        similarity_input_embedding = F.cosine_similarity(
            original_embedding["input_embedding"],
            special_embedding["input_embedding"],
            dim=2  # Compare along the embedding dimension
        )

        similarity_output_embedding = F.cosine_similarity(
            original_embedding["output_embedding"],
            special_embedding["output_embedding"],
            dim=2  # Compare along the embedding dimension
        )

        print(f"\nOriginal: {original_phrase}")
        print(f"Special:  {special_phrase}")
        print(f"Input embedding: Are the embeddings exactly the same? {torch.allclose(original_embedding['input_embedding'], special_embedding['input_embedding'])}")
        print("Input embedding: Cosine similarity for each token:")
        for i, sim in enumerate(similarity_input_embedding[0]):
            print(f"  Token {i+1}: {sim.item():.4f}")
        print(f"Input embedding: Average similarity: {similarity_input_embedding.mean().item():.4f}")

        print(f"\nOutput embedding: Are the embeddings exactly the same? {torch.allclose(original_embedding['output_embedding'], special_embedding['output_embedding'])}")
        print("Output embedding: Cosine similarity for each token:")
        for i, sim in enumerate(similarity_output_embedding[0]):
            print(f"  Token {i+1}: {sim.item():.4f}")
        print(f"Output embedding: Average similarity: {similarity_output_embedding.mean().item():.4f}")

        print("\n" + "-"*50)
