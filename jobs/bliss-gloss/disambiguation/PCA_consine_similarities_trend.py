# python PCA_consine_similarities_trend.py <output_file_path>
# python PCA_consine_similarities_trend.py ./test_results/

'''
Calculates the cosine similarity between the direct average of embeddings and the
average of their principal components. The average is computed iteratively by
including an increasing number of principal components, starting from the first
up to all available components. The results are saved in a CSV file for further plot.
'''

import csv
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_input_embedding import get_embeddings_of_words


def calculate_pca_average_similarities_full(embeddings):
    """
    Calculates the cosine similarity between the direct average of embeddings and the
    average of their top 'i' principal components, iterating through all components.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (N, D) where N is the number
                                   of samples and D is the feature dimension.

    Returns:
        list: A list of D float values, where the element at index k corresponds to the
              cosine similarity using the average of PC1 through PC-(k+1).
    """
    n_samples, n_features = embeddings.shape
    print(f"Number of samples: {n_samples}, Number of features: {n_features}")
    if n_samples < 2:
        raise ValueError("At least two embeddings are required for PCA.")

    # 1. Calculate the direct average embedding (our reference)
    direct_average_embedding = torch.mean(embeddings, dim=0)

    # 2. Perform PCA to get the sorted principal components
    centered_embeddings = embeddings - torch.mean(embeddings, dim=0)
    covariance_matrix = torch.cov(centered_embeddings.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvectors (principal components) in descending order of importance
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_principal_components = eigenvectors[:, sorted_indices]

    # 3. Iterate from i=1 up to the total number of features/eigenvectors
    similarities = []
    # The loop now goes from 1 to n_features (inclusive)
    for i in range(1, n_features + 1):
        # Select the top 'i' principal components
        top_i_pcs = sorted_principal_components[:, :i]

        # Calculate the average of these selected components
        average_from_pcs = torch.mean(top_i_pcs, dim=1)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(direct_average_embedding, average_from_pcs, dim=0, eps=1e-8)

        # Store the Python float value
        similarities.append(similarity.item())

    return similarities


if len(sys.argv) != 2:
    print("Usage: python PCA_cosine_similarities_trend.py <output_file_path>")
    sys.exit(1)

output_file_path = sys.argv[1]

target_tokens = [" break", " fracture", " injury", " damage"]
symbol_id = 24852

# target_tokens = [" floor", " level"]
# symbol_id = 23085

# target_tokens = [" attachment", " joint", " seam", " appendix", " annex"]
# symbol_id = 23409

# target_tokens = [" dispersion", " dissemination", " scattering", " spread", " spreading"]
# symbol_id = 24887

model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to(device)
model.eval()

tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)
embeddings = get_embeddings_of_words(model, tokenizer, target_tokens)

# Calculate the similarities
cosine_similarities = calculate_pca_average_similarities_full(embeddings)

# Save the results to a CSV file
output_filename = os.path.expanduser(os.path.join(output_file_path, f"PCA_cosine_similarities_trend_{symbol_id}.csv"))

with open(output_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["num_components", "cosine_similarity"])  # Write a header row
    for i, sim in enumerate(cosine_similarities):
        writer.writerow([i + 1, sim])
print(f"Successfully saved similarity trend to {output_filename}")
