
# This script is to use attention-weighted embeddings for Bliss glosses that are composed by
# multiple tokens. It computes token important scores based on attention weights for each token
# in the gloss, calculates weighted input and output embeddings and assigns them to the new
# token for that Bliss symbol. In the end, it evaluates the weighted embeddings by comparing prediction
# probabilities between the original gloss and the Bliss token in the same prefix context.
#
# Usage: python weighted_embedding_for_multi_token_gloss.py
#
# ==== Results with the gloss "alligator" ====
#  {'token_importance_scores': {'Ġall': 0.7225201725959778, 'igator': 0.27747979760169983}}
# ==================
# * Prefix Sentence: I saw an
# Bliss token Rank: 41402
# Original first token rank: 9803
# Probability ratio with first token: 0.029093955536499263
# Probability ratio with joint: 0.39177466989197546
# Top k predictions: [(' article', 0.09820226579904556), (' ad', 0.07555928826332092), (' old', 0.0742218866944313), (' interview', 0.036730796098709106), (' opportunity', 0.03522808849811554)]
# Bliss token probability: 1.5451583124104218e-08
# Original token probabilities: [5.310925530466193e-07, 0.07426196336746216]
# ==================
# * Prefix Sentence: The fierce
# Bliss token Rank: 20931
# Original first token rank: 30617
# Probability ratio with first token: 3.671008990324822
# Probability ratio with joint: 223.16988217314946
# Top k predictions: [(' and', 0.05123656988143921), (' battle', 0.03826843202114105), (' debate', 0.033074572682380676), (' competition', 0.027975283563137054), (' determination', 0.01796548254787922)]
# Bliss token probability: 1.1517551001816173e-06
# Original token probabilities: [3.137434703148756e-07, 0.016449391841888428]
# ==================
# * Prefix Sentence: Watch out for the
# Bliss token Rank: 11068
# Original first token rank: 25410
# Probability ratio with first token: 7.299363689401912
# Probability ratio with joint: 211.1666354783328
# Top k predictions: [(' following', 0.020335126668214798), (' signs', 0.0047517498023808), (' little', 0.0044301459565758705), (' "', 0.004428068175911903), (' slippery', 0.0038461247459053993)]
# Bliss token probability: 1.2175501979072578e-05
# Original token probabilities: [1.6680223779985681e-06, 0.03456684201955795]
# ==================
# * Prefix Sentence: In the swamp lives an
# Bliss token Rank: 36249
# Original first token rank: 2171
# Probability ratio with first token: 0.001883876639474093
# Probability ratio with joint: 0.002103809708239897
# Top k predictions: [(' all', 0.19014839828014374), (' old', 0.08003611117601395), (' ancient', 0.07036222517490387), (' evil', 0.03705969080328941), (' animal', 0.025520160794258118)]
# Bliss token probability: 2.8162590481883853e-08
# Original token probabilities: [1.4949275282560848e-05, 0.8954595923423767]
# ==================
# Number of tests: 4
# Average probablity ratio with the first original token: 2.750337627975677
# Average probablity ratio with the joint tokens: 108.68259903277063

import os
import torch
import torch.nn.functional as F
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class AttentionWeightExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_attention_weights(self, text: str) -> torch.Tensor:
        """
        Extract attention weights with robust handling of matrix shapes.
        """
        # Tokenize input text
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokens["input_ids"][0]

        # Get model's attention patterns
        with torch.no_grad():
            outputs = self.model(
                **tokens,
                output_attentions=True,  # Request attention matrices
            )

        # Investigate and print attention matrices shape for debugging
        attention_matrices = torch.stack(outputs.attentions)

        # Handle different matrix shapes
        if len(attention_matrices.shape) == 5:
            # Typical shape: (layers, batch, heads, seq_len, seq_len)
            # Average across layers, batch, and heads
            avg_attention = attention_matrices.mean(dim=(0, 1, 2))
        elif len(attention_matrices.shape) == 4:
            # Some models might have shape (batch, heads, seq_len, seq_len)
            avg_attention = attention_matrices.mean(dim=(0, 1))
        elif len(attention_matrices.shape) == 3:
            # If already averaged or in a different format
            avg_attention = attention_matrices.mean(dim=0)
        else:
            raise ValueError(f"Unexpected attention matrix shape: {attention_matrices.shape}")

        return avg_attention, token_ids

    def get_token_importance(self, text: str) -> Dict[str, float]:
        """
        Calculate importance scores for each token based on attention patterns.
        """
        attention_matrix, token_ids = self.get_attention_weights(text)

        # Ensure attention_matrix is 2D
        if len(attention_matrix.shape) > 2:
            attention_matrix = attention_matrix.squeeze()

        # Verify matrix is square and matches token length
        if attention_matrix.shape[0] != len(token_ids):
            print(f"Warning: Attention matrix shape {attention_matrix.shape} "
                  f"doesn't match token length {len(token_ids)}")
            # Truncate or pad as needed
            min_len = min(attention_matrix.shape[0], len(token_ids))
            attention_matrix = attention_matrix[:min_len, :min_len]
            token_ids = token_ids[:min_len]

        try:
            # Incoming attention (columns)
            incoming_attention = attention_matrix.mean(dim=0)
            # Outgoing attention (rows)
            outgoing_attention = attention_matrix.mean(dim=1)
        except RuntimeError as e:
            print(f"Error calculating attention: {e}")
            print(f"Attention matrix shape: {attention_matrix.shape}")
            raise
        print(f"get_token_importance(): Attention matrices: {attention_matrix}\nincoming_attention: {incoming_attention}\noutgoing_attention: {outgoing_attention}")

        # Combine incoming and outgoing attention
        combined_scores = (incoming_attention + outgoing_attention) / 2

        # Create token to score mapping
        token_scores = {}
        for idx, token_id in enumerate(token_ids):
            token = self.tokenizer.convert_ids_to_tokens(token_id.item())
            token_scores[token] = combined_scores[idx].item()

        return token_scores

    def get_weighted_embedding(self, text: str) -> torch.Tensor:
        """
        Get weighted embedding for a multi-token word based on attention scores.
        """
        # Get token scores
        token_scores = self.get_token_importance(text)

        # Get token embeddings
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokens["input_ids"][0]
        input_embeddings = self.model.get_input_embeddings().weight[token_ids]
        output_embeddings = self.model.get_input_embeddings().weight[token_ids]

        # Convert scores to weights using softmax
        weights = torch.tensor([token_scores[self.tokenizer.convert_ids_to_tokens(id.item())] for id in token_ids])
        weights = F.softmax(weights, dim=0)

        # Apply weights to embeddings
        weighted_input_embedding = (input_embeddings * weights.unsqueeze(1)).sum(dim=0)
        weighted_input_embedding = (output_embeddings * weights.unsqueeze(1)).sum(dim=0)

        return weighted_input_embedding, weighted_input_embedding, token_scores


class BlissEmbeddingHandlerWithAttention:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.attention_extractor = AttentionWeightExtractor(self.model, self.tokenizer)
        self.bliss_to_token_map = {}

    def add_bliss_symbol(self, bliss_token: str, gloss: str) -> Dict[str, float]:
        """
        Add a new Blissymbol token to the model's vocabulary using attention-weighted embeddings.
        Returns the token importance scores for analysis.
        """
        # Get weighted input/output embedding and token scores
        weighted_input_embedding, weighted_output_embedding, token_scores = self.attention_extractor.get_weighted_embedding(gloss)

        # Add new token to tokenizer
        self.tokenizer.add_tokens([bliss_token])
        new_token_id = self.tokenizer.convert_tokens_to_ids(bliss_token)

        # Resize embedding matrices
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Assign weighted embedding to new token
        with torch.no_grad():
            self.model.get_input_embeddings().weight[new_token_id] = weighted_input_embedding
            # For output embedding, we could use the same weighted embedding
            # or experiment with different combinations
            self.model.get_output_embeddings().weight[new_token_id] = weighted_output_embedding

        self.bliss_to_token_map[bliss_token] = new_token_id

        return token_scores

    def analyze_token_weights(self, gloss: str) -> Dict[str, any]:
        """
        Analyze how attention weights are distributed across tokens.
        """
        token_scores = self.attention_extractor.get_token_importance(gloss)

        analysis = {
            "token_scores": token_scores
        }

        return analysis


class EmbeddingEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compare_predictions(self, prefix_text: str, original_word: str, bliss_token: str) -> dict:
        """
        Compare prediction probabilities between original word and Bliss token
        given the same prefix context.

        Args:
            prefix_text: The context text before the target word
            original_word: The original word (e.g., "alligator")
            bliss_token: The Bliss token (e.g., "[BLISS_ALLIGATOR]")

        Returns:
            Dictionary containing comparison results
        """
        # Encode the prefix text
        prefix_encoding = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**prefix_encoding)
            logits = outputs.logits[:, -1, :]  # Get logits for next token prediction
            probs = F.softmax(logits, dim=-1)

        # Get token IDs for both versions
        original_tokens = self.tokenizer(original_word, add_special_tokens=False)["input_ids"]
        bliss_token_id = self.tokenizer.convert_tokens_to_ids(bliss_token)

        # Get probability for Bliss token
        bliss_prob = probs[0, bliss_token_id].item()

        # Calculate joint probability for original tokens
        original_probs = []
        current_prefix = prefix_text
        for token_id in original_tokens:
            # Get prediction for next token
            current_encoding = self.tokenizer(current_prefix, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                outputs = self.model(**current_encoding)
                token_logits = outputs.logits[:, -1, :]
                token_probs = F.softmax(token_logits, dim=-1)
                original_probs.append(token_probs[0, token_id].item())
            # Update prefix for next token prediction
            current_prefix += self.tokenizer.decode([token_id])

        # Calculate joint probability (multiply individual probabilities)
        original_joint_prob = torch.tensor(original_probs).prod().item()

        # Get top predictions for context analysis
        top_k = 5
        top_probs, top_indices = probs[0].topk(top_k)
        top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
        top_predictions = list(zip(top_tokens, top_probs.tolist()))

        return {
            "bliss_token_probability": bliss_prob,
            "original_token_probabilities": original_probs,
            "original_joint_probability": original_joint_prob,
            "probability_ratio_with_first_token": bliss_prob / original_probs[0] if original_probs[0] > 0 else float('inf'),
            "probability_ratio_with_joint": bliss_prob / original_joint_prob if original_joint_prob > 0 else float('inf'),
            "top_k_predictions": top_predictions,
            "bliss_token_rank": (probs[0] >= bliss_prob).sum().item(),
            "original_first_token_rank": (probs[0] >= original_probs[0]).sum().item() if original_probs else None
        }


# ['Ġall', 'igator']
gloss = " alligator"
new_bliss_token = "[BLISS_ALLIGATOR]"

model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize the handler
handler = BlissEmbeddingHandlerWithAttention(model, tokenizer)

# Add a multi-token Bliss symbol and get token importance scores
token_scores = handler.add_bliss_symbol(new_bliss_token, gloss)

# Analyze the attention weights
analysis = handler.analyze_token_weights(gloss)
print("Token importance analysis:\n", analysis)

# Evaluation: Compare prediction probabilities between the original word
# and the Bliss token in the same prefix context.
evaluator = EmbeddingEvaluator(model, tokenizer)

# Define test cases with different contexts
test_cases = [
    ("I saw an", "alligator", "[BLISS_ALLIGATOR]"),
    ("The fierce", "alligator", "[BLISS_ALLIGATOR]"),
    ("Watch out for the", "alligator", "[BLISS_ALLIGATOR]"),
    ("In the swamp lives an", "alligator", "[BLISS_ALLIGATOR]")
]

results = []

for prefix, original, bliss in test_cases:
    result = evaluator.compare_predictions(prefix, original, bliss)
    results.append({
        "prefix": prefix,
        "metrics": result
    })

    print("==================")
    print(f"* Prefix Sentence: {prefix}")
    print(f"Bliss token Rank: {result['bliss_token_rank']}")
    print(f"Original first token rank: {result['original_first_token_rank']}")
    print(f"Probability ratio with first token: {result['probability_ratio_with_first_token']}")
    print(f"Probability ratio with joint: {result['probability_ratio_with_joint']}")
    print(f"Top k predictions: {result['top_k_predictions']}")
    print(f"Bliss token probability: {result['bliss_token_probability']}")
    print(f"Original token probabilities: {result['original_token_probabilities']}")

# Aggregate results
avg_prob_ratio_with_first_token = sum(r["metrics"]["probability_ratio_with_first_token"] for r in results) / len(results)
avg_prob_ratio_with_joint = sum(r["metrics"]["probability_ratio_with_joint"] for r in results) / len(results)
avg_rank_diff = sum(abs(r["metrics"]["bliss_token_rank"] - r["metrics"]["original_first_token_rank"])
                    for r in results if r["metrics"]["original_first_token_rank"] is not None) / len(results)

print("==================")
print(f"Number of tests: {len(results)}")
print(f"Average probablity ratio with the first original token: {avg_prob_ratio_with_first_token}")
print(f"Average probablity ratio with the joint tokens: {avg_prob_ratio_with_joint}")
