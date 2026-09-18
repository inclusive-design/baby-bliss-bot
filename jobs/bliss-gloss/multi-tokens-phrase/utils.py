# Utility functions for running the disambiguation scripts that caculates the output embedding
# of the new token that captures the disambiguated meaning of an English word.

import torch


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
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context)
        hidden_states.append(h)
        target_logits.append(-10)  # discourage predicting the target token

    return torch.cat(hidden_states, dim=0).to(model.device), torch.tensor(target_logits, device=model.device)


def calc_output_embedding(hidden_states, target_logits, dtype=None):
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


# Get the average input embedding for a list of multi-token words/phrases
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


# Get the contextual embedding of the last token in the phrase when the phrase is appended to each of the prefix sentences
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


# Calculate loss for a given prefix sentence and target hidden state
def calculate_loss(model, tokenizer, prefix_sentence, learnable_input_embedding, target_hidden_state, loss_fn, device):
    prefix_inputs = tokenizer(prefix_sentence, return_tensors="pt").to(device)
    prefix_embeddings = model.get_input_embeddings()(prefix_inputs.input_ids)

    # Concatenate the prefix embeddings with the learnable embedding
    inputs_embeds = torch.cat([prefix_embeddings, learnable_input_embedding], dim=1)

    attention_mask = torch.cat([
        prefix_inputs.attention_mask,
        torch.ones(1, 1, device=device, dtype=torch.long)
    ], dim=1)

    # Pass `inputs_embeds` directly to the model
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
    current_hidden_state = outputs.hidden_states[-1][:, -1, :]

    target = torch.ones(current_hidden_state.shape[0]).to(device)
    loss = loss_fn(current_hidden_state, target_hidden_state, target)
    return loss


def optimize_input_embedding(model, tokenizer, epochs, learning_rate, phrase, training_positive_context_sentences, validation_positive_context_sentences, device):
    """Optimization of input embedding for a multi-tokens phrase."""
    print("Starting optimization of the input embedding...")
    model.eval()
    target_hidden_states = get_target_contextual_embedding(model, tokenizer, training_positive_context_sentences, phrase)
    validation_hidden_states = get_target_contextual_embedding(model, tokenizer, validation_positive_context_sentences, phrase)

    # Freeze all model parameters since we are only optimizing the input embedding in a variable
    print("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the input embedding layer specifically which allows the forward pass to build a computation graph.
    # This is necessary to work around the error of "RuntimeError: element 0 of tensors does not require grad and
    # does not have a grad_fn" if all model parameters are frozen. However, Since the optimizer STILL ONLY targets
    # the learnable input embedding, `loss.backward()` will calculate gradients for the embedding layer, but the
    # optimizer will ignore them, so the input embedding layer will not be updated.
    model.get_input_embeddings().weight.requires_grad = True

    # Use the average input embedding of all composing tokens in the phrase as the
    # initial embedding for the optimization
    unsqueezed_average_input_embedding = get_average_input_embeddings(model, tokenizer, [phrase]).unsqueeze(0).unsqueeze(0).to(device)
    learnable_input_embedding = torch.nn.Parameter(unsqueezed_average_input_embedding, requires_grad=True)

    # Optimization
    optimizer = torch.optim.AdamW(
        [learnable_input_embedding],
        lr=learning_rate
    )
    loss_fn = torch.nn.CosineEmbeddingLoss()

    # Track the best validation loss and the best input embedding
    best_validation_loss = float('inf')
    best_input_embedding = None

    for epoch in range(epochs):
        # Set the model to training mode. The model will be switched to eval mode during validation.
        model.train()
        total_loss = 0
        # Iterate through the prefixes and their pre-calculated target hidden states
        for prefix_sentence, target_hidden_state in target_hidden_states.items():
            optimizer.zero_grad()

            loss = calculate_loss(model, tokenizer, prefix_sentence, learnable_input_embedding, target_hidden_state, loss_fn, device)

            # Backpropagate and Update
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(target_hidden_states)

        # Validation after every epoch
        model.eval()
        total_validation_loss = 0

        # Wrap the validation loop with torch.no_grad()
        with torch.no_grad():
            for prefix_sentence, target_hidden_state in validation_hidden_states.items():
                validation_loss = calculate_loss(model, tokenizer, prefix_sentence, learnable_input_embedding, target_hidden_state, loss_fn, device)
                total_validation_loss += validation_loss.item()

        avg_validation_loss = total_validation_loss / len(validation_hidden_states)

        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.6f}, Average Validation Loss: {avg_validation_loss:.6f}")

        # Save the best input embedding if the validation loss has improved
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            best_input_embedding = learnable_input_embedding.detach().clone()
            print(f"**  New best validation loss: {best_validation_loss:.6f}.")

    print(f"\nOptimization finished! Best validation loss: {best_validation_loss:.6f}\n")
    return best_input_embedding.squeeze()


def add_token_to_model(model, tokenizer, new_token, input_emb, output_emb):
    """Add new token to model's vocabulary and embedding matrices."""
    # Add token to tokenizer
    num_added_tokens = tokenizer.add_tokens([new_token])
    if num_added_tokens == 0:
        raise ValueError("Token already exists in vocabulary")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)

    # Set the embeddings
    with torch.no_grad():
        model.get_input_embeddings().weight[new_token_id] = input_emb
        model.get_output_embeddings().weight[new_token_id] = output_emb

    return new_token_id


def get_rank_for_tokens(all_ranks, target_token_ids):
    # Retrieve ranks for target token IDs
    target_ranks = []
    for token_id in target_token_ids:
        rank_position = (all_ranks == token_id).nonzero().item()
        target_ranks.append(rank_position + 1)  # Convert to 1-based ranking
    return target_ranks


def test_token_prediction(model, tokenizer, context_sentences, new_token_id, target_token_ids):
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
            rank_of_new_token_id = get_rank_for_tokens(sorted_indices, [new_token_id])
            rank_of_target_token_ids = get_rank_for_tokens(sorted_indices, target_token_ids)

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'rank_of_new_token_id': rank_of_new_token_id,  # Convert to 1-based ranking
                'rank_of_target_token_ids': rank_of_target_token_ids,  # Convert to 1-based ranking
                'top_5_predictions': top_5_tokens
            })

    return results


def get_token_prediction(model, tokenizer, context_sentences, target_token_ids):
    """Test the ranks of the target tokens in predictions for each testing context."""
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

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'rank_of_target_token_ids': get_rank_for_tokens(sorted_indices, target_token_ids),  # Now a list of ranks
                'top_5_predictions': top_5_tokens
            })

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


def evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, phrase
):
    # Test token prediction
    results = test_token_prediction(model, tokenizer, training_positive_context_sentences, new_token_id, target_token_ids)
    print_results("Re-verify predictions on POSITIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, training_negative_context_sentences, new_token_id, target_token_ids)
    print_results("Re-verify predictions on NEGATIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, testing_positive_context_sentences, new_token_id, target_token_ids)
    print_results("Predictions on TESTING POSITIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, testing_negative_context_sentences, new_token_id, target_token_ids)
    print_results("Predictions on TESTING NEGATIVE training sentences:", target_tokens, new_token, results)

    # Text Generation
    print("\nValidation - Generation:")

    for prompt_template in testing_text_generation_prompts:
        print(f"Prompt: {prompt_template}")

        if (isinstance(phrase, list)):
            for p in phrase:
                original_prompt = prompt_template.format(placeholder=p)
                print(f"Generated text with {p}: {generate_text(model, tokenizer, original_prompt)}")
        else:
            original_prompt = prompt_template.format(placeholder=phrase)
            print(f"Generated text with {phrase}: {generate_text(model, tokenizer, original_prompt)}")

        new_prompt = prompt_template.format(placeholder=new_token)
        print(f"Generated text with {new_token}: {generate_text(model, tokenizer, new_prompt)}\n")
