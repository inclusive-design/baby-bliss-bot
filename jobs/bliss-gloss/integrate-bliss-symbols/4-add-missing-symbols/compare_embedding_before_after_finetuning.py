import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir_before_finetuning = os.path.expanduser("~/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/3_model_12356")
model_dir_after_finetuning = os.path.expanduser("~/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/4_model_12341")  # Where you saved your LoRA model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir_after_finetuning)

# 1. Load Original Model's Embeddings
# This might involve loading the base model and then getting its embeddings
print("Loading original model...")
model_before_finetuning = AutoModelForCausalLM.from_pretrained(
    model_dir_before_finetuning,
    torch_dtype=torch.bfloat16  # or torch.float16 depending on your setup
)
input_embeddings_before = model_before_finetuning.get_input_embeddings().weight.data.clone().to(device)
output_embeddings_before = model_before_finetuning.get_output_embeddings().weight.data.clone().to(device)
print(f"Embeddings shape before fine-tuning: {input_embeddings_before.shape}")
del model_before_finetuning  # Free up memory
torch.cuda.empty_cache()

# 2. Load Fine-Tuned Model's Embeddings
# This typically involves loading the base model and then applying the LoRA adapter
print("Loading fine-tuned model...")
model_after_finetuning = AutoModelForCausalLM.from_pretrained(
    model_dir_after_finetuning,
    torch_dtype=torch.bfloat16
)
input_embeddings_after = model_after_finetuning.get_input_embeddings().weight.data.clone().to(device)
output_embeddings_after = model_after_finetuning.get_output_embeddings().weight.data.clone().to(device)
print(f"Embeddings shape after fine-tuning: {input_embeddings_after.shape}")

del model_after_finetuning  # Free up memory
torch.cuda.empty_cache()


# Get the vocabulary size (number of tokens)
vocab_size = input_embeddings_before.shape[0]

# Prepare lists to store results
token_ids = []
tokens = []
input_embedding_cosine_similarities = []
input_embedding_euclidean_distances = []
output_embedding_cosine_similarities = []
output_embedding_euclidean_distances = []


def calculate_distances(vec_before_finetuning, vec_after_finetuning):
    # Calculate Cosine Similarity
    cos_sim = torch.nn.functional.cosine_similarity(vec_before_finetuning.unsqueeze(0), vec_after_finetuning.unsqueeze(0)).item()

    # Calculate Euclidean Distance
    eu_dist = torch.norm(vec_before_finetuning - vec_after_finetuning).item()

    return cos_sim, eu_dist


print("Calculating changes for all tokens...")
for i in range(vocab_size):
    token_ids.append(i)

    # Decode token ID to actual token string.
    try:
        token_str = tokenizer.decode([i])
    except Exception:
        print(f"Error decoding token ID: {i}")
        token_str = f"[UNK_ID_{i}]"  # Fallback for special/unknown IDs
    tokens.append(token_str)

    input_embedding_cos_sim, input_embedding_eu_dist = calculate_distances(input_embeddings_before[i], input_embeddings_after[i])
    input_embedding_cosine_similarities.append(input_embedding_cos_sim)
    input_embedding_euclidean_distances.append(input_embedding_eu_dist)

    output_embedding_cos_sim, output_embedding_eu_dist = calculate_distances(output_embeddings_before[i], output_embeddings_after[i])
    output_embedding_cosine_similarities.append(output_embedding_cos_sim)
    output_embedding_euclidean_distances.append(output_embedding_eu_dist)

# Create a DataFrame for easy analysis
pd.set_option("display.float_format", lambda x: "%.4f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

results_df = pd.DataFrame({
    "token_id": token_ids,
    "token": tokens,
    "input_embedding_cosine_similarity": input_embedding_cosine_similarities,
    "input_embedding_euclidean_distance": input_embedding_euclidean_distances,
    "output_embedding_cosine_similarity": output_embedding_cosine_similarities,
    "output_embedding_euclidean_distance": output_embedding_euclidean_distances
})

filtered_df = results_df[~results_df["token"].str.startswith("<|reserved_special_token_", na=False)]

# Sort by Euclidean distance to see the most changed tokens
results_df_sorted_input_embedding_euclidean = filtered_df.sort_values(by="input_embedding_euclidean_distance", ascending=False)

# Sort by Cosine Similarity to see tokens whose direction changed the most (lower cosine similarity)
results_df_sorted_input_embedding_cosine = filtered_df.sort_values(by="input_embedding_cosine_similarity", ascending=True)

print("\nInput Embedding - Top 20 tokens with largest Euclidean distance change:")
print(results_df_sorted_input_embedding_euclidean.head(20))

print("\nInput Embedding - Top 20 tokens with lowest cosine similarity:")
print(results_df_sorted_input_embedding_cosine.head(20))

# Sort by Euclidean distance to see the most changed tokens
results_df_sorted_output_embedding_euclidean = filtered_df.sort_values(by="output_embedding_euclidean_distance", ascending=False)

# Sort by Cosine Similarity to see tokens whose direction changed the most (lower cosine similarity)
results_df_sorted_output_embedding_cosine = filtered_df.sort_values(by="output_embedding_cosine_similarity", ascending=True)

print("\nOutput Embedding - Top 20 tokens with largest Euclidean distance change:")
print(results_df_sorted_output_embedding_euclidean.head(20))

print("\nOutput Embedding - Top 20 tokens with lowest cosine similarity:")
print(results_df_sorted_output_embedding_cosine.head(20))
