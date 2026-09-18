# Usage:
# python ~/bliss_gloss/add_bliss_tokens.py ~/bliss_gloss/data/bliss_gloss_cleaned_synonyms.json ~/bliss_gloss/outputs/models/llama-first-single-token-8B ~/bliss_gloss/outputs/bliss_ids_added.json ~/bliss_gloss/outputs/bliss_ids_not_added.json

import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utility_funcs import get_first_single_token_embeddings, get_mean_single_token_embeddings

# When True, assign output embeddings to new Bliss tokens so the model can output new Bliss tokens.
# When False, do not assign output embeddings to new tokens so the output layer is not aware of new tokens
# and the model only outputs English.
ASSIGN_BLISS_OUTPUT_EMBEDDING = True

# When True, zero out all old output embeddings and only keep the new Bliss embeddings
# so the model will only output Bliss tokens.
# When False, keep both old and new output embeddings so the model will output a mix of English and Bliss.
ZERO_OUT_OLD_OUTPUT_EMBEDDING = False

# When True, assign the new Bliss token with input and output embeddings of the first single-token gloss.
# When False, assign the mean input and output embeddings of all single-token glosses for each Bliss symbol.
USE_FIRST_SINGLE_TOKEN_GLOSS = True

if len(sys.argv) != 5:
    print("Usage: python add_bliss_tokens.py <input_gloss_json> <output_model_path> <output_file_with_added_id> <output_file_with_not_added_id>")
    sys.exit(1)

input_gloss_json = sys.argv[1]
output_model_path = sys.argv[2]
output_file_with_added_id = sys.argv[3]
output_file_with_not_added_id = sys.argv[4]

# Load the JSON file
with open(input_gloss_json, 'r') as f:
    input_gloss_data = json.load(f)

# Load the local Llama model
# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)


not_added_bliss_ids = {}
added_bliss_ids = {}
new_tokens_to_add = []
new_token_input_embeddings = []
new_token_output_embeddings = []

# Process each record in the JSON
for bliss_id, glosses in input_gloss_data.items():
    calculated_glosses, input_emb, output_emb = (get_first_single_token_embeddings if USE_FIRST_SINGLE_TOKEN_GLOSS else get_mean_single_token_embeddings)(model, tokenizer, glosses)

    if calculated_glosses:
        special_token = f"[BLISS_{bliss_id}]"
        new_tokens_to_add.append(special_token)
        new_token_input_embeddings.append(input_emb)
        if ASSIGN_BLISS_OUTPUT_EMBEDDING:
            new_token_output_embeddings.append(output_emb)
        added_bliss_ids[bliss_id] = calculated_glosses
    else:
        not_added_bliss_ids[bliss_id] = glosses

# Add all new tokens at once
num_added_bliss_ids = tokenizer.add_tokens(new_tokens_to_add)

# Resize token embeddings once
model.resize_token_embeddings(len(tokenizer))

# Update embeddings for new tokens
with torch.no_grad():
    if ZERO_OUT_OLD_OUTPUT_EMBEDDING:
        # Zero out all tokens in the output layer
        model.lm_head.weight.data.fill_(0)

    # Add new Bliss tokens
    for i, new_token in enumerate(new_tokens_to_add):
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        model.get_input_embeddings().weight[new_token_id] = new_token_input_embeddings[i]
        if ASSIGN_BLISS_OUTPUT_EMBEDDING:
            model.lm_head.weight[new_token_id] = new_token_output_embeddings[i]

print(f"{len(added_bliss_ids)} are added.")
print(f"{len(not_added_bliss_ids)} are not added.")

# Save the updated model
print("Saving updated model...")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("Model saved...")

# Save added tokens to file
with open(output_file_with_added_id, 'w') as f:
    json.dump(added_bliss_ids, f, indent=2)

# Save not added tokens to file
with open(output_file_with_not_added_id, 'w') as f:
    json.dump(not_added_bliss_ids, f, indent=2)


# Return next word predictions with probability greater than a threshold
def predict_next_words_above_threshold(sentence, threshold=0.1, max_predictions=10):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

    # Generate logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the probabilities for the next token
    next_token_logits = logits[:, -1, :]  # Only the last token is relevant
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Filter tokens with probability above the threshold
    top_tokens = torch.where(probabilities > threshold)[1]
    top_probs = probabilities[0, top_tokens]

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]

    # Sort by probability
    top_predictions = sorted(zip(top_words, top_probs), key=lambda x: -x[1])

    return top_predictions[:max_predictions]


# Return top_k number of next word predictions
def predict_top_k_words(sentence, top_k=10):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Generate logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the probabilities for the next token
    next_token_logits = logits[:, -1, :]  # Only the last token is relevant
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top n tokens with highest probabilities
    top_probs, top_tokens = torch.topk(probabilities, top_k)

    # Convert token ids back to words
    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens[0]]

    top_predictions = list(zip(top_words, top_probs[0]))

    return top_predictions


# Convert bag of words into a sentence using the model. This function runs for GPU.
def generate_text_with_prompt(prompt, model, tokenizer, temperature=0.7):
    print(f"Prompt: {prompt}\n")

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Move input_ids to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=temperature
    )
    return tokenizer.batch_decode(outputs.cpu().detach().numpy(), skip_special_tokens=True)[0][len(prompt):]


def convert_bag_of_words(model, tokenizer, bag_of_words):
    # Create a prompt to guide the model
    prompt = f"Create a complete sentence from the following words: {bag_of_words}."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the sentence with the model
    output_tokens = model.generate(
        input_ids=inputs['input_ids'],
        max_length=100,  # limit the length of generated text
        num_beams=5,  # use beam search for better results
        no_repeat_ngram_size=2,  # prevent repetitive phrases
        early_stopping=True  # stop when a sentence is formed
    )

    # Decode the output tokens back to a string
    generated_sentence = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_sentence


sentences = [
    "I'm hungry I want to",
    "[BLISS_14916][BLISS_12639][BLISS_14912][BLISS_14916][BLISS_18035][BLISS_17739]",
    "I'm hungry so I want to",
    "[BLISS_14916][BLISS_12639][BLISS_14912][BLISS_17705][BLISS_14916][BLISS_18035][BLISS_17739]",
    "I would like to eat some delicious",
    "[BLISS_14916][BLISS_24264][BLISS_15210][BLISS_17739][BLISS_13906][BLISS_17207][BLISS_22352]",
    "The doctor said it's important to",
    "[BLISS_13861][BLISS_16728][BLISS_14960][BLISS_12639][BLISS_14930][BLISS_17739]",
    "She carefully placed the fragile",
    "[BLISS_14688][BLISS_13123][BLISS_16440][BLISS_17700][BLISS_12897]"
]  # Incomplete sentence to predict

top_k = 6  # Number of top predictions to return

for sentence in sentences:
    print(f"Predicted sentence: {sentence}")

    predictions = predict_top_k_words(sentence, top_k)

    print(f"Top {top_k} predictions:")
    for word, prob in predictions:
        print(f"{word}: {prob.item():.4f}")

# threshold = 0.1  # 10% probability threshold
# predictions = predict_next_words_above_threshold(probabilities, threshold)

# print(f"\nPredictions above threshold {threshold}:")
# for word, prob in predictions:
#     print(f"{word}: {prob.item():.4f}\n")

bag_of_words = ["nice weather walk", "[BLISS_15717][BLISS_18214][BLISS_18031]",
                "caregiver action keep home clean", "[BLISS_23063][BLISS_23007][BLISS_15143][BLISS_14885][BLISS_24062]"]

for words in bag_of_words:
    prompt = f"The given bag of words are from an AAC user who expresses himself telegraphically.\
 Please help to convert what the user said to first-person sentences. Only respond with converted sentences: {words}."
    print(f"A bag of word: {words}")
    print(f"Converted sentence: {generate_text_with_prompt(prompt, model, tokenizer)}\n")
