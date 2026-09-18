# Utility functions for running add_symbols.py
import os
import torch
import re
import random


###################################################
# START of utility functions for building the dataset

# Replace special characters in a sentence. This is a must have for sentences in all datasets
# because LLMs tend to use "’" instead of "'".
def replace_special_chars(sentence):
    return sentence.replace("’", "'")


# Replace the whole word in a sentence with a given replacement
# Note: target_words can be a word string or a list of string words
def replace_whole_word(text, target_words, replace_with):
    if isinstance(target_words, str):
        target_words = [target_words]
    for target_word in target_words:
        pattern = r"\s*\b{}\b".format(re.escape(target_word))
        text = re.sub(pattern, replace_with, text, flags=re.IGNORECASE)
    return text


# Truncate the sentence to the last occurrence of any gloss
def get_partial_sentences_by_keywords(sentence, glosses):
    all_indices = []

    for gloss in glosses:
        if not gloss.strip():  # Skip empty or whitespace-only glosses
            continue

        # Match whole word with word boundaries and escape special characters
        pattern = r'\b{}\b'.format(re.escape(gloss))

        # Find all matches and their start positions
        for match in re.finditer(pattern, sentence):
            all_indices.append(match.start())

    if not all_indices:
        print(f"Error: No gloss found in sentence: '{sentence}'")
        return None

    last_phrase_start = max(all_indices)
    truncated = sentence[:last_phrase_start]
    return truncated.rstrip()


# Process positive context sentences
# 1. Replace special characters: "’" -> "'"
# 2. Truncate the sentence to the last occurrence of any gloss
# 3. Remove sentences that are too short (less than MIN_PARTIAL_SENTENCE_LENGTH)
# 4. Split the sentences into training and testing sets (80% training, 20% testing)
def process_positive_context_sentences(model, tokenizer, orig_positive_context_sentences, glosses, target_token_ids, min_partial_sentence_length, rank_threshold):
    # Process all sentences first
    partial_sentences = []
    for sentence in orig_positive_context_sentences:
        sentence = replace_special_chars(sentence)
        partial_sentence = get_partial_sentences_by_keywords(sentence, glosses)
        if partial_sentence is not None and len(partial_sentence) > min_partial_sentence_length:
            partial_sentences.append(partial_sentence)

    sentences_with_prediction_ranks = test_token_prediction(model, tokenizer, partial_sentences, target_token_ids)

    final_sentences = []
    for result in sentences_with_prediction_ranks:
        # At least one of the target token IDs should have a rank below the threshold
        if any(int(rank) < rank_threshold for rank in result["rank_of_target_token_ids"]):
            final_sentences.append(result["context"])

    # Calculate split indices
    total_sentences = len(final_sentences)
    training_size = int(total_sentences * 0.8)  # 80% for training

    return final_sentences[:training_size], final_sentences[training_size:]


# Process negative context sentences
# 1. Replace special characters: "’" -> "'"
# 2. Split the sentences into training and testing sets (80% training, 20% testing)
def process_negative_context_sentences(model, tokenizer, orig_negative_context_sentences, target_token_ids, rank_threshold):
    # Replace special characters in the original sentences
    interim_sentences = [replace_special_chars(sentence) for sentence in orig_negative_context_sentences]

    sentences_with_prediction_ranks = test_token_prediction(model, tokenizer, interim_sentences, target_token_ids)

    final_sentences = []
    for result in sentences_with_prediction_ranks:
        # At least one of the target token IDs should have a rank below the threshold
        if all(int(rank) > rank_threshold for rank in result["rank_of_target_token_ids"]):
            final_sentences.append(result["context"])

    # Calculate split indices
    total_sentences = len(final_sentences)
    training_size = int(total_sentences * 0.8)  # 80% for training

    return final_sentences[:training_size], final_sentences[training_size:]


# Process fine-tuning sentences
# 1. Replace special characters: "’" -> "'"
# 2. Replace glosses of the in-processing bci_av_id with the corresponding token. These sentences compose
#    the dataset for fine-tuning.
# 3. Go through the list of BCI-AV-IDs already in the model and replace their glosses with the corresponding tokens.
#    These sentences are appended to the fine-tuning dataset to ensure the model learns how different Bliss tokens
#    interact with each other.
# 4. Randomize the order of the sentences to ensure a good mix of training and validation data.
# 5. Split the sentences into training and validation sets (80% training, 20% validation)
def process_fine_tuning_sentences(orig_fine_tuning_sentences, bci_av_id, token_template, existing_id_gloss_list):
    # Replace special characters in the original sentences
    normalized_sentences = [replace_special_chars(sentence) for sentence in orig_fine_tuning_sentences]

    # Replace glosses of the in-processing bci_av_id with the corresponding token
    glosses_for_current_id = existing_id_gloss_list[bci_av_id]
    processed_single_sentences = [replace_whole_word(sentence, glosses_for_current_id, " " + token_template.format(bciAvId=bci_av_id)) for sentence in normalized_sentences]

    # Go through the list of BCI-AV-IDs already in the model and replace their glosses with the corresponding tokens
    # This is to ensure the fine-tuning learns how different Bliss tokens interact with each other
    processed_sentences = []
    for sentence in processed_single_sentences:
        modified = sentence
        for gloss_id, target_glosses in existing_id_gloss_list.items():
            modified = replace_whole_word(modified, target_glosses, " " + token_template.format(bciAvId=gloss_id))
        if modified != sentence:
            processed_sentences.append(modified)

    # Merge the processed sentences with the ones that only glosses for the in-processing bci_av_id were replaced
    processed_sentences.extend(processed_single_sentences)

    # Randomnize the order of the sentences to ensure a good mix of training and validation data
    random.shuffle(processed_sentences)

    # Calculate split indices
    total_sentences = len(processed_sentences)
    fine_tuning_size = int(total_sentences * 0.8)  # 80% for training

    return processed_sentences[:fine_tuning_size], processed_sentences[fine_tuning_size:]


# Process testing text generation prompts:
# 1. Replace special characters in the original sentences
# 2. Remove the empty space before the placeholder in the testing text generation prompts
def process_testing_text_generation_prompts(orig_testing_text_generation_prompts):
    # Replace special characters in the original sentences
    interim_prompts = [replace_special_chars(prompt) for prompt in orig_testing_text_generation_prompts]
    # Remove the empty space before the placeholder
    return [prompt.replace(" {placeholder}", "{placeholder}") for prompt in interim_prompts]


# The format of the output file is: <output_dir>/dataset_<bci_av_id>_<glosses>.py
# The glosses are processed to remove all non-alphanumeric characters and joined with underscores.
def get_output_file_location(output_dir, bci_av_id, glosses):
    # Remove all non-alphanumeric characters from each gloss
    processed_glosses = [re.sub(r'[^a-zA-Z0-9]', '', gloss) for gloss in glosses]

    # Join processed glosses with underscores
    gloss_part = '_'.join(processed_glosses)

    # Construct the full file path
    return os.path.join(output_dir, f"dataset_{bci_av_id}_{gloss_part}.py")

# END of utility functions for building the dataset
###################################################


###################################################
# START of utility functions for adding a new symtol token to the model
def get_hidden_state_and_next_token_logits(model, tokenizer, text, return_logits=False):
    """Get the hidden states before final layer for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Extract the hidden state of the last token in the sequence from the last layer
        hidden_state_from_last_layer = outputs.hidden_states[-1][:, -1, :]
        # Extract the logit for the next token prediction
        next_token_logits = outputs.logits[:, -1, :] if return_logits else None
    return hidden_state_from_last_layer, next_token_logits


def create_training_data(model, tokenizer, positive_context_sentences, negative_context_sentences, target_token_ids):
    """Create training data from context sentences."""

    # Convert target token ids to list if not already
    target_token_ids = target_token_ids if isinstance(target_token_ids, list) else [target_token_ids]

    hidden_states = []
    target_logits = []

    # Positive context sentences - high target logits
    for context in positive_context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context, True)
        hidden_states.append(h)
        target_logits.append(logits[0, target_token_ids].max().item())

    # Negative context sentences - low target logits
    for context in negative_context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context, True)
        hidden_states.append(h)
        target_logits.append(logits[0, target_token_ids].min().item())  # discourage predicting the target token

    return torch.cat(hidden_states, dim=0).to(model.device), torch.tensor(target_logits, device=model.device)


def calc_embeddings(hidden_states, target_logits, dtype=None):
    # Set the minimal safe dtype for lstsq as default
    lstsq_dtype = torch.float32  # default for stability/speed balance

    # Override to user's dtype ONLY if it's float32/float64
    if dtype in (torch.float32, torch.float64):
        lstsq_dtype = dtype

    # Convert inputs to lstsq-compatible dtype
    hidden_states = hidden_states.to(lstsq_dtype)
    target_logits = target_logits.to(lstsq_dtype)

    # Compute solution in higher precision
    output_emb = torch.linalg.lstsq(hidden_states, target_logits.unsqueeze(1)).solution.squeeze(1)

    # Cast result to user's dtype
    if dtype is not None and dtype != lstsq_dtype:
        output_emb = output_emb.to(dtype)

    return output_emb


def calc_output_embedding(model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids, dtype):

    hidden_states, target_logits = create_training_data(
        model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
    )
    return calc_embeddings(hidden_states, target_logits, dtype)


def get_average_input_embeddings(model, tokenizer, words):
    """Compute average input embeddings for a list of multi-token words/phrases"""
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer.weight.data

    word_vectors = []

    for word in words:
        tokens = tokenizer.tokenize(word)
        if not tokens:  # Handle edge case for empty tokens
            continue

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Get embeddings for all sub-tokens
        token_embeddings = embeddings[token_ids]

        # Average across sub-tokens for this word
        word_embedding = torch.mean(token_embeddings, dim=0)
        word_vectors.append(word_embedding)

    # Average across all words
    if not word_vectors:  # Handle empty input case
        return None

    return torch.mean(torch.stack(word_vectors), dim=0)


def get_rank_for_tokens(all_ranks, target_token_ids):
    # Retrieve ranks for target token IDs
    target_ranks = []
    for token_id in target_token_ids:
        rank_position = (all_ranks == token_id).nonzero().item()
        target_ranks.append(rank_position + 1)  # Convert to 1-based ranking
    return target_ranks


def test_token_prediction(model, tokenizer, context_sentences, target_token_ids, new_token_id=None):
    """Test the rank of new token in predictions for each testing context."""

    # Convert target token ids to list if not already
    target_token_ids = target_token_ids if isinstance(target_token_ids, list) else [target_token_ids]

    results = []

    for context in context_sentences:
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Get token rankings
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            if new_token_id is not None:
                rank_of_new_token_id = get_rank_for_tokens(sorted_indices, [new_token_id])
            rank_of_target_token_ids = get_rank_for_tokens(sorted_indices, target_token_ids)

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            result = {
                'context': context,
                'rank_of_target_token_ids': rank_of_target_token_ids,
                'top_5_predictions': top_5_tokens
            }

            if new_token_id is not None:
                result["rank_of_new_token_id"] = rank_of_new_token_id

            results.append(result)

    return results


def print_results(title, target_tokens, new_token, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_tokens}: {result['rank_of_target_token_ids']}")
        if new_token:
            print(f"Rank of {new_token}: {result['rank_of_new_token_id'][0]}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_text_generation(
    model, tokenizer, testing_text_generation_prompts, new_token, phrase
):
    for prompt_template in testing_text_generation_prompts:
        print(f"Prompt: {prompt_template}")

        if isinstance(phrase, list):
            for p in phrase:
                original_prompt = prompt_template.format(placeholder=p)
                print(f"Generated text with {p}: {generate_text(model, tokenizer, original_prompt)}")
        else:
            original_prompt = prompt_template.format(placeholder=phrase)
            print(f"Generated text with {phrase}: {generate_text(model, tokenizer, original_prompt)}")

        new_prompt = prompt_template.format(placeholder=new_token)
        print(f"Generated text with {new_token}: {generate_text(model, tokenizer, new_prompt)}\n")


def evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, phrase
):
    # Test token prediction
    results = test_token_prediction(model, tokenizer, training_positive_context_sentences, target_token_ids, new_token_id)
    print_results("Re-verify predictions on POSITIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, training_negative_context_sentences, target_token_ids, new_token_id)
    print_results("Re-verify predictions on NEGATIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, testing_positive_context_sentences, target_token_ids, new_token_id)
    print_results("Predictions on TESTING POSITIVE testing sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, testing_negative_context_sentences, target_token_ids, new_token_id)
    print_results("Predictions on TESTING NEGATIVE testing sentences:", target_tokens, new_token, results)

    # Text Generation
    print("\nValidation - Generation:")
    test_text_generation(model, tokenizer, testing_text_generation_prompts, new_token, phrase)

# END of utility functions for adding a new symbol token to the model
###################################################
