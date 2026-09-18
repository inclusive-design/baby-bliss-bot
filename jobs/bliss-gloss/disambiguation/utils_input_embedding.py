# Utility functions for running the disambiguation scripts that caculates the output embedding
# of the new token that captures the disambiguated meaning of an English word.

import torch
from kneed import KneeLocator
import torch.nn.functional as F


# Functions for calculating output embedding
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
        # target_logits.append(-10)  # discourage predicting the target token
        target_logits.append(logits[0, target_token_ids].min().item())

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


# Get embeddings for a list of words/phrases. If a word/phrase contains multiple tokens,
# average the embeddings of all composing tokens.
def get_embeddings_of_words(model, tokenizer, words, type="input_embeddings"):
    if (type != "input_embeddings") and (type != "output_embeddings"):
        raise ValueError("type must be either 'input_embeddings' or 'output_embeddings'")

    embedding_layer = model.get_input_embeddings() if type == "input_embeddings" else model.get_output_embeddings()
    embeddings = embedding_layer.weight.data

    word_vectors = []

    for word in words:
        tokens = tokenizer.tokenize(word)
        if not tokens:  # Handle edge case for empty tokens
            continue

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_embeddings = embeddings[token_ids]

        # Average across sub-tokens for this word
        word_embedding = torch.mean(token_embeddings, dim=0)
        word_vectors.append(word_embedding)

    return torch.stack(word_vectors) if len(word_vectors) > 0 else None


# Get the average input embedding for a list of multi-token words/phrases
def get_average_input_embeddings(model, tokenizer, words):
    """Compute average input embeddings for a list of multi-token words/phrases"""
    word_vectors = get_embeddings_of_words(model, tokenizer, words, type="input_embeddings")
    return torch.mean(word_vectors, dim=0) if word_vectors is not None else None


# Get the average output embedding for a list of multi-token words/phrases
def get_average_output_embeddings(model, tokenizer, words):
    """Compute average output embeddings for a list of multi-token words/phrases"""
    word_vectors = get_embeddings_of_words(model, tokenizer, words, type="output_embeddings")
    return torch.mean(word_vectors, dim=0) if word_vectors is not None else None


def get_embedding_by_PCA_kneed(synonym_embeddings, kneed_sensitivity=1):
    """
    Generates a new token by projecting the average embedding onto a subspace
    spanned by the optimal number of principal components. The optimal number is
    determined automatically by finding the "elbow" of the scree plot using the
    kneed library.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (N, D) where N is the number
                                    of synonym embeddings and D is the embedding dimension.
        kneed_sensitivity (float): Sensitivity parameter for the KneeLocator. Higher values
                                    detect elbows more aggressively. Default is 1.

    Returns:
        - projected_average (torch.Tensor): The average of the synonym embeddings
                                            projected onto the subspace of the top N PCs.
    """
    # 1. Calculate Covariance Matrix
    # Center the data by subtracting the mean
    n_samples = synonym_embeddings.shape[0]
    print("Number of synonym embeddings:", n_samples)
    average_embedding = torch.mean(synonym_embeddings, dim=0)
    centered_embeddings = synonym_embeddings - average_embedding

    # Calculate the covariance matrix = (X^T @ X) / (n-1)
    covariance_matrix = torch.matmul(centered_embeddings.T, centered_embeddings) / (n_samples - 1)

    # 2. Find Principal Components (Eigenvectors)
    # torch.linalg.eigh() returns eigenvalues in ascending order, so need to reverse them.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # 3. Select Top N Components based on Explained Variance
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 4. Automatically Find the Elbow using kneed
    num_meaningful_components = n_samples - 1

    if num_meaningful_components < 3:
        # kneed needs at least 3 points. If we have fewer, just take the first PC.
        n_components = 1
        print("Warning: Less than 4 embeddings provided. Defaulting to n_components=1.")
    else:
        # Slice the eigenvalues to only include the meaningful ones
        meaningful_eigenvalues = sorted_eigenvalues[:num_meaningful_components]

        kneedle = KneeLocator(
            x=range(1, len(meaningful_eigenvalues) + 1),
            y=meaningful_eigenvalues.cpu().numpy(),
            S=kneed_sensitivity,
            curve="concave",
            direction="decreasing"
        )

        n_components = kneedle.elbow

        if n_components is None:
            print("Warning: kneed could not find an elbow within meaningful components. Defaulting to 1.")
            n_components = 1

    # Get the top N principal components
    top_n_pcs = sorted_eigenvectors[:, :n_components]

    # 5. Project the Average Embedding
    # The projection of the average_embedding vector onto the subspace spanned by top_n_pcs is:
    # P @ P.T @ average_embedding
    # where P is the matrix of principal components.
    # Project the average embedding onto the principal component subspace
    # This is a linear combination of the principal components, where the coefficients
    # are the dot products of the average embedding with each principal component.
    projected_average = torch.matmul(top_n_pcs, torch.matmul(top_n_pcs.T, average_embedding))

    if projected_average is not None:
        print(f"\nNumber of principal components retained to kneed sensitivity {kneed_sensitivity}: {n_components}")
        print(f"Shape of the top N PCs matrix: {top_n_pcs.shape if top_n_pcs is not None else 'N/A'}")
        print(f"Shape of the final projected average vector: {projected_average.shape}")
        print(f"\nProjected Average Tensor (using top {n_components} PCs): {projected_average}\n")

    return projected_average


def get_embedding_by_PCA_threshold(synonym_embeddings, explained_variance_threshold=0.95):
    # 1. Calculate Covariance Matrix
    # Center the data by subtracting the mean
    n_samples = synonym_embeddings.shape[0]
    print("Number of synonym embeddings:", n_samples)
    average_embedding = torch.mean(synonym_embeddings, dim=0)
    centered_embeddings = synonym_embeddings - average_embedding

    # Calculate the covariance matrix = (X^T @ X) / (n-1)
    covariance_matrix = torch.matmul(centered_embeddings.T, centered_embeddings) / (n_samples - 1)

    # 2. Find Principal Components (Eigenvectors)
    # torch.linalg.eigh() returns eigenvalues in ascending order, so need to reverse them.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # 3. Select Top N Components based on Explained Variance
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the cumulative explained variance
    total_variance = torch.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    cumulative_explained_variance = torch.cumsum(explained_variance_ratio, dim=0)

    # Determine the number of components that meet the threshold
    # Use searchsorted to find the first index where the cumulative sum exceeds the threshold.
    n_components = torch.searchsorted(cumulative_explained_variance, explained_variance_threshold).item() + 1

    # Ensure n_components is not larger than the number of available components
    n_components = min(n_components, len(sorted_eigenvalues))

    # Get the top N principal components
    top_n_pcs = sorted_eigenvectors[:, :n_components]

    # 4. Project the Average Embedding
    # The projection of the average_embedding vector onto the subspace spanned by top_n_pcs is:
    # P @ P.T @ average_embedding
    # where P is the matrix of principal components.
    # Project the average embedding onto the principal component subspace
    # This is a linear combination of the principal components, where the coefficients
    # are the dot products of the average embedding with each principal component.
    projected_average = torch.matmul(top_n_pcs, torch.matmul(top_n_pcs.T, average_embedding))

    if projected_average is not None:
        print(f"\nNumber of principal components retained to explain {explained_variance_threshold*100}% of variance: {n_components}")
        print(f"Shape of the top N PCs matrix: {top_n_pcs.shape if top_n_pcs is not None else 'N/A'}")
        print(f"Shape of the final projected average vector: {projected_average.shape}")
        print(f"\nProjected Average Tensor (using top {n_components} PCs): {projected_average}\n")

    return projected_average


def get_embedding_by_PC1_and_variance_contributions(embeddings):
    """
    Calculates the primary principal component (PC1) and returns the variance contribution
    percentage of each of the top N-1 components.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (N, D) where N is the number
                                   of synonym embeddings and D is the embedding dimension.

    Returns:
        tuple: A tuple containing:
            - pc1 (torch.Tensor): The primary principal component vector (shape [D]).
            - variance_percentages (torch.Tensor): A 1D tensor of length N-1, where each
                                                    element is the variance percentage
                                                    contributed by the corresponding PC.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("At least two embeddings are required for PCA.")

    # 1. Perform PCA
    centered_embeddings = embeddings - torch.mean(embeddings, dim=0)
    covariance_matrix = torch.cov(centered_embeddings.T)

    # Use eigh for symmetric matrices. it's faster and more stable.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 2. Get the Primary Principal Component (PC1)
    # PC1 is the eigenvector corresponding to the largest eigenvalue.
    pc1 = sorted_eigenvectors[:, 0]

    # 3. Calculate Variance Contribution Percentages
    # The total variance is the sum of all eigenvalues.
    total_variance = torch.sum(sorted_eigenvalues)

    # Only care about the N-1 meaningful components
    num_meaningful_components = n_samples - 1
    meaningful_eigenvalues = sorted_eigenvalues[:num_meaningful_components]

    variance_percentages = torch.tensor([])  # Default empty tensor
    if total_variance > 0:
        # Calculate percentage for each of the top N-1 components
        variance_percentages = (meaningful_eigenvalues / total_variance) * 100
    else:
        print("Warning: Total variance is zero. All contributions are zero.")
        variance_percentages = torch.zeros(num_meaningful_components)

    return pc1, variance_percentages


def get_embedding_by_average_PCs(embeddings):
    """
    Calculates the average of meaningful principal components (top N-1) and the weighted sum
    of meaningful principal components based on their variance contribution percentages.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (N, D) where N is the number
                                   of synonym embeddings and D is the embedding dimension.

    Returns:
        tuple: A tuple containing:
            - average_of_eigenvectors (torch.Tensor): The average of the meaningful principal
            components (shape [D]).
            - weighted_sum_of_eigenvectors (torch.Tensor): The weighted sum of the meaningful
            principal components (shape [D]).
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("At least two embeddings are required for PCA.")

    # 1. Perform PCA
    centered_embeddings = embeddings - torch.mean(embeddings, dim=0)
    print(f"shape of centered embeddings: {centered_embeddings.shape}")
    covariance_matrix = torch.cov(centered_embeddings.T)
    print(f"shape of covariance matrix: {covariance_matrix.shape}")

    # Use eigh for symmetric matrices. it's faster and more stable.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    print(f"Top 10 Sorted Eigenvalues: {sorted_eigenvalues[:10]}")

    # 3. Calculate Variance Contribution Percentages
    # The total variance is the sum of all eigenvalues.
    total_variance = torch.sum(sorted_eigenvalues)

    # Only care about the N-1 meaningful components
    num_meaningful_components = n_samples - 1
    meaningful_eigenvalues = sorted_eigenvalues[:num_meaningful_components]
    meaningful_eigenvectors = sorted_eigenvectors[:, :num_meaningful_components]

    print(f"Meaningful Eigenvalues: {meaningful_eigenvalues}")

    if total_variance > 0:
        # Calculate percentage for each of the top N-1 components
        weights = meaningful_eigenvalues / total_variance
    else:
        print("Warning: Total variance is zero. All contributions are zero.")
        weights = torch.zeros(num_meaningful_components)

    average_of_eigenvectors = torch.mean(meaningful_eigenvectors, dim=1)

    # This equivalent to weighted sum of the eigenvectors:
    # sum(eigenvector_i * weight_i for i in range(num_meaningful_components))
    weighted_sum_of_eigenvectors = meaningful_eigenvectors @ weights

    return average_of_eigenvectors, weighted_sum_of_eigenvectors


def get_embedding_by_self_attention(synonym_embeddings):
    """
    Calculates an unambiguous embedding for a cluster of synonyms using an attention mechanism.

    Args:
        synonym_embeddings: A 2D tensor where each row is an embedding of a synonym.

    Returns:
        A 1D tensor representing the aggregated, unambiguous embedding.
    """

    # 1. Define a "Query": Calculate the average of the synonym embeddings.
    # This represents the "general idea" of the cluster.
    query_vector = torch.mean(synonym_embeddings, dim=0)

    # 2. Calculate Scores: Compute the dot product of each synonym embedding with the query vector.
    scores = torch.matmul(synonym_embeddings, query_vector)

    # 3. Normalize Scores (Softmax): Apply a softmax function to the scores
    # to get a probability distribution of attention weights.
    attention_weights = F.softmax(scores, dim=0)
    print(f"Attention weights: {attention_weights}")

    # 4. Compute Weighted Sum: Calculate the weighted sum of the original
    # synonym embeddings using the attention weights.
    # We unsqueeze the attention_weights to make it a column vector for broadcasting.
    unambiguous_embedding = torch.sum(attention_weights.unsqueeze(1) * synonym_embeddings, dim=0)

    return unambiguous_embedding


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


# Functions for evaluation
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
                print(f"* Generated text with '{p}': {generate_text(model, tokenizer, original_prompt)}")
        else:
            original_prompt = prompt_template.format(placeholder=phrase)
            print(f"* Generated text with '{phrase}': {generate_text(model, tokenizer, original_prompt)}")

        new_prompt = prompt_template.format(placeholder=new_token)
        print(f"** Generated text with '{new_token}': {generate_text(model, tokenizer, new_prompt)}\n")
