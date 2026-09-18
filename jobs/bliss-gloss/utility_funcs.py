import torch
import torch.nn.functional as F


def preprocess_gloss(gloss):
    return f" {gloss.replace(' ', '').replace('-', '')}"


# Calculate if two embeddings are equal and the cosine similarity between two embeddings
def compare_embedding(emb1, emb2):
    return {
        "are_equal": torch.allclose(emb1, emb2, atol=1e-6),
        "similarity": round(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item(), 4)
    }


# Get the contextual embedding of a phrase
def get_contextual_embedding(model, tokenizer, phrase):
    inputs = tokenizer(phrase, return_tensors="pt", add_special_tokens=False)
    # inputs = tokenizer(phrase, return_tensors="pt")

    # Pass the tokens through the model to get the contextual embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states[-1].squeeze(0)


# Function to get embeddings for a specific word in a phrase
def get_word_embedding_in_phrase(model, tokenizer, phrase, word):
    # Tokenize the phrase
    inputs = tokenizer(phrase, return_tensors="pt")

    # Get the token ID for the target word
    word_token_id = tokenizer.encode(f" {word}", add_special_tokens=False)[0]
    print(f"Token ID for '{word}' in phrase '{phrase}': {word_token_id}")

    # Find the position of the word in the tokenized input
    word_position = (inputs.input_ids == word_token_id).nonzero()[0, 1]
    print(f"Word position for '{word} in phrase '{phrase}': {word_position}")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the input embeddings of the phrase
    input_embeddings = model.get_input_embeddings()(inputs['input_ids'])

    return {
        "input_embedding": input_embeddings[0, word_position],
        "contextual_embedding": outputs.hidden_states[-1][0, word_position, :]
    }


# Loop through the given glosses and return the first gloss that can be converted into
# a single token, along with its input and output embeddings. If none of the glosses
# can be converted to a single token, return (None, None, None)
# Return: a tuple of (first_single_token_gloss, input_embedding, output_embedding)
def get_first_single_token_embeddings(model, tokenizer, glosses):
    single_token_gloss = None
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_id) == 1:
            single_token_gloss = gloss
            break

    if single_token_gloss:
        return single_token_gloss, model.get_input_embeddings().weight[token_id], model.lm_head.weight[token_id]
    else:
        return None, None, None


# Loop through the given glosses, find all glosses that are single tokens, calculate
# the mean input and output embeddings of these single token glosses, and return the
# mean embeddings. If none of the glosses are single tokens, return (None, None, None)
# Return: a tuple of (list_of_all_single_token_glosses, mean_input_embedding, mean_output_embedding)
def get_mean_single_token_embeddings(model, tokenizer, glosses):
    single_token_glosses = []
    input_embeddings = []
    output_embeddings = []

    # Loop through each gloss to find its token id and input/output embedding
    for gloss in glosses:
        gloss = preprocess_gloss(gloss)
        tokens = tokenizer.tokenize(gloss)
        token_id = tokenizer.convert_tokens_to_ids(tokens)

        # Check if it's a single token gloss
        if len(token_id) > 1:
            continue

        # Get input and output embedding of the token
        input_embedding = model.get_input_embeddings().weight[token_id]  # the input embedding
        output_embedding = model.lm_head.weight[token_id]  # the output embedding

        single_token_glosses.append(gloss)
        input_embeddings.append(input_embedding)
        output_embeddings.append(output_embedding)

    # Calculate the mean input and output embeddings of all single token glosses
    if len(single_token_glosses) > 0:
        mean_input_embedding = torch.mean(torch.stack(input_embeddings), dim=0)
        mean_output_embedding = torch.mean(torch.stack(output_embeddings), dim=0)

        return single_token_glosses, mean_input_embedding, mean_output_embedding
    else:
        return None, None, None
