# Usage: python output_embedding_optimize.py <epochs> <learning_rate>

"""
This script adds a new special token to a llama model to represent specifically
the meaning of an animal bark, disambiguating it from other meanings like tree bark.
This script creates a training dataset from two sets of context sentences: one set
of positive training sentences that may predict the word "bark" in the sense of an
animal bark. Another set of negative training sentences that may either predict the
word "bark" in the sense of tree bark or not predict the word "bark" in any sense as
a top prediction. The script then uses these two datasets to optimize the input and
output embeddings for the new token.

This script has 3 versions in its commit history:
V1 - train both the input embedding and output embedding only with positive
training context sentences. The similarity between the input embedding and the output
embedding is considered in the loss function.
* Training result:
    - The loss is decreasing over epochs
    - The new token is effective in predicting "bark" in the sense of an animal bark
    - The new token is not effective in predicting "bark" in the sense of tree bark.
      The new token ranks the number 1 in a couple of tree bark context sentences
      when the expected rank should be very low.
    - The new token is not effective in predicting "bark" in random context sentences.
      The new token ranks the number 1 in a couple of sentences when the expected rank
      should be very low.

V2 - train both the input embedding and output embedding with both positive and negative
training context sentences. The similarity between the input embedding and the output
embedding is considered in the loss function.
    - The loss is decreasing over epochs
    - The new token is effective in predicting "bark" in random context sentences.
    - The new token is not effective in predicting "bark" in the sense of dog bark.
      The new token ranks low when the expected rank should be very high.
    - The new token is not effective in predicting "bark" in the sense of tree bark.
      The new token ranks the number 1 in a couple of tree bark context sentences
      when the expected rank should be very low.

V3 - train only the output embedding with both positive and negative training. The
input embedding is set to the original "bark" token's input embedding. The similarity
between the input embedding and the output embedding is not considered as the input
embedding is fixed.
    - The loss reaches 0 after over 100 epochs
    - The new token is effective in predicting "bark" in random context sentences.
    - The new token is not effective in predicting "bark" in the sense of dog bark.
      The new token ranks low when the expected rank should be very high.
    - The new token is not effective in predicting "bark" in the sense of tree bark.
      The new token ranks the number 1 in a couple of tree bark context sentences
      when the expected rank should be very low.

The script does:
1. Collecting hidden states output from the last layer and logits from context sentences that may
   predict "bark" in the sense of an animal bark
2. Using these to optimize both input and output embeddings for the new token through:
   - Output embedding optimization: Making the token predict similarly to "bark" in positive contexts
   - Joint loss function that balances prediction accuracy and embedding consistency

The script then verifies the effectiveness of the training of the new token by checking the
prediction rank of the token " bark" and the new token in all context sentences for training

The script then tests the effectiveness of the new token by:
- Checking the prediction rank of the token " bark" in context sentences for testing
- Checking the prediction rank of the new token in the same context sentences
- Checking top 5 predicted tokens

Note: This approach helps create more semantically precise tokens by learning from contextual
usage patterns rather than directly copying existing token embeddings.
"""

import sys
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_output_embedding import create_training_data, optimize_embeddings, add_token_to_model, test_token_prediction


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


if len(sys.argv) != 3:
    print("Usage: python output_embedding_optimize.py <epochs> <learning_rate>")
    sys.exit(1)

epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])

# Positive context sentences for training that would predict "bark" in the sense of an animal bark.
# They capture different aspects that lead to a potential prediction the word "bark" meaning
# an animal/dog bark. Half of them are about animals. The other half are not in the
# context of dogs, but may still predict "bark" in the sense of an animal bark.
training_positive_context_sentences = [
    "The excited puppy began to",
    "When strangers approach, guard dogs will",
    "Hearing a noise outside, the dog started to",
    "Seeing the mailman, the German Shepherd would always",
    "To alert its owner of danger, a dog will",
    "The frightened dog let out a loud",
    "Feeling threatened, the small dog would",
    "Playfully, the energetic puppy would",
    "Late at night, the neighborhood dogs often",
    "During the full moon, wolves howl and dogs",
    "At the sight of a squirrel, the terrier would",
    "While chasing cats, dogs typically",
    "To communicate with other dogs, a puppy will",
    "To get attention from its owner, the dog would",
    "Rather than whimper, the dog decided to",
    "The eerie silence of the forest was occasionally broken by the distant, sharp",
    "Even quiet breeds sometimes need to",
    "The sudden noise echoed through the neighborhood - a deep, threatening",
    "At midnight, an unfamiliar sound made me jump: a sharp, piercing",
    "From somewhere in the darkness came a loud, aggressive",
    "Behind the fence, I heard an angry",
    "She was about to fall asleep when a loud, unexpected sound broke the silence — a sharp",
    "The sound echoed through the yard, a sharp and sudden",
    "From the corner of the garden came a loud, cheerful",
    "The silence was broken by a deep and resounding",
    "In the distance, I heard an excited"
]

# Negative context sentences for training that would NOT predict "bark" in the sense of an animal bark.
# Half of them are about trees that may predict "bark" in the sense of tree bark. The other half are
# random sentences that may not predict "bark" in any sense.
training_negative_context_sentences = [
    # In the sense of dog bark
    "The well-trained dog quietly",   # emphasizing silence
    "Looking at the tree's rough",    # different meaning of bark
    "After the walk, the tired dog",  # tired state
    "While petting the calm dog",     # peaceful state
    # Different meaning of "bark" in the sense of tree bark
    "The herbal tea was made from dried",
    "As she climbed the tree, her hands scraped against the jagged",
    "She carefully stripped the",
    "He painted a beautiful landscape, adding fine details to the tree",
    "He admired the intricate carvings etched into the",
    "The loud crackling sound came as the fire consumed the dry",
    "The craftsman used strips of",
    "She traced her fingers over the initials carved into the tree’s",
    "He loved the rich texture of the",
    "The recipe called for a pinch of powdered",
    # Random sentences
    "The stars shimmered brightly against the velvety night",
    "She carefully folded the letter and placed it in the drawer, her heart heavy with unspoken",
    "The train whistle echoed through the valley, signaling its arrival at the small",
    "The aroma of freshly baked bread filled the air, drawing people into the cozy",
    "The sun dipped below the horizon, casting a warm glow over the",
    "The gentle rustling of leaves filled the quiet forest as the sun set behind the",
    "He couldn’t decide between the red sweater and the blue",
    "After weeks of planning, the team finally launched their new",
    "The aroma of freshly baked bread wafted through the tiny",
    "The glass vase shattered into a thousand pieces when it slipped from her"
]

# Context sentences for testing, generated by ChatGPT 3.5.
# The first 5 sentences tend to lead to a prediction of "bark" in the sense of an animal bark.
# The middle 5 sentences tend to lead to a prediction of "bark" in the sense of tree bark.
# The last 5 sentences are random sentences that may not predict "bark" in any sense.
testing_context_sentences = [
    "As soon as the stranger stepped into the yard, the dog let out a sharp",
    "The trainer used a clicker to stop the dog from continuing to",
    "I was walking down the quiet street when, out of nowhere, I heard a sudden",
    "The calm evening was interrupted by the echo of a distant",
    "She paused, straining to listen, and then the unmistakable",
    "The forest was dense and quiet, with towering trees whose rough texture was most noticeable on their thick trunks, covered in deep, cracked",
    "The young explorers marveled at the tall trees, each one displaying unique features, such as the coarse and rugged",
    "She paused to examine the large tree, noting how its surface was rough and worn, with the thick layers of",
    "Walking along the trail, he ran his hand over the tree's surface, feeling the texture shift as he brushed against the weathered",
    "As we ventured deeper into the woods, we saw that many of the trees had a distinct pattern on their trunks, a textured surface that resembled hardened",
    "The gentle rustling of leaves filled the quiet forest as the sun set behind the",
    "He couldn’t decide between the red sweater and the blue",
    "After weeks of planning, the team finally launched their new",
    "The aroma of freshly baked bread wafted through the tiny",
    "The sun dipped below the horizon, casting a warm glow over the"
]

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

# Optimize embeddings
embed_dim = model.config.hidden_size
output_emb = optimize_embeddings(
    model, hidden_states, target_logits, embed_dim, epochs, learning_rate
)

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
