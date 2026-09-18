# Usage: python output_embedding_calc.py

"""
This script adds a new special token to a llama model to represent specifically
the meaning of an animal bark, disambiguating it from other meanings like tree bark.
This script creates a training dataset from two sets of context sentences: one set
of positive training sentences that may predict the word "bark" in the sense of an
animal bark. Another set of negative training sentences that may either predict the
word "bark" in the sense of tree bark or not predict the word "bark" in any sense as
a top prediction. From these context sentences, the script collects the contextual
embedding (CE), which is the hidden state output from the last layer, and the logits for
the next token (L). Assuming the output embedding of the new token is OE, the formula
for these three variables is: CE . OE = L.

Assuming the number of context sentences is N, and the model's embedding dimension is D,
the tensor holding all collected contextual embedding (T_CE) is of shape (N, D) while
the tensor holding all collected logits (T_L) is of shape (N, 1). To perform the dot
product across all context sentences, this matrix multiplication is used: T_CE * OE = T_L.

To calculate OE, the script inverts the matrix multiplication using torch.linalg.lstsq()
function.

The input embedding of the new token is set to the input embedding of the original token
"bark". The script then adds the new token to the model's vocabulary and sets the output
embedding of the new token to the calculated output embedding.

The script then verifies the effectiveness of the training of the new token by checking
the prediction rank of the token "bark" and the new token in all context sentences for
training and testing. The test result can be found in the
"test_results/V4-result_calc_output_embedding.log".
"""

import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import data_animal_bark
from utils_output_embedding import create_training_data, calc_embeddings, add_token_to_model, test_token_prediction


def print_results(title, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_token}: {result['rank_of_target_token_id']}")
        print(f"Rank of {new_token}: {result['rank_of_new_token_id']}")
        print(f"Rank difference: {result['rank_of_target_token_id'] - result['rank_of_new_token_id']}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


training_positive_context_sentences = data_animal_bark.training_positive_context_sentences
training_negative_context_sentences = data_animal_bark.training_negative_context_sentences
testing_context_sentences = data_animal_bark.testing_context_sentences

# Track the total running time of this script
start_time = time.time()

# Load model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Set the model to evaluation mode
model.eval()

# Get original "bark" token id for training data
target_token = " bark"
tokens = tokenizer.tokenize(target_token)
bark_token_id = tokenizer.convert_tokens_to_ids(tokens)[0]
print("bark_token_id:", bark_token_id)

# Get the input embedding of the target token
target_token_input_embedding = model.get_input_embeddings().weight[bark_token_id]

# Prepare training data
hidden_states, target_logits = create_training_data(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, bark_token_id
)

# Calculate output embedding by inverting the matrix multiplication with least squares problem
output_emb = calc_embeddings(hidden_states, target_logits)

# Add new token to model
new_token = "[BLISS_24020]"
new_token_id = add_token_to_model(model, tokenizer, new_token, target_token_input_embedding, output_emb)

end_time_training = time.time()
elapsed_time = end_time_training - start_time
print(f"Execution time for training: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Re-verify predictions on training sentences for the new token
results = test_token_prediction(model, tokenizer, training_positive_context_sentences, new_token_id, bark_token_id)
print_results("Re-verify predictions on POSITIVE training sentences:", results)

results = test_token_prediction(model, tokenizer, training_negative_context_sentences, new_token_id, bark_token_id)
print_results("Re-verify predictions on NEGATIVE training sentences:", results)

# Test predictions
results = test_token_prediction(model, tokenizer, testing_context_sentences, new_token_id, bark_token_id)
print_results("Predictions on TESTING context sentences:", results)

end_time = time.time()

# Calculate elapsed time
end_time_testing = time.time()
elapsed_time = end_time_testing - end_time_training
print(f"Execution time for testing: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
