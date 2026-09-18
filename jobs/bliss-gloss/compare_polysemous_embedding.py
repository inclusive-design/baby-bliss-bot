# Compare input embeddings and contextual embeddings of the same word in difference sentences(contexts)
# 1. The input embedding to the first layer of the neural netword
# 2. The contextual embedding from the last layer of the hidden layer

import os
from transformers import LlamaTokenizer, LlamaModel
from utility_funcs import compare_embedding, get_word_embedding_in_phrase

# The local directory with the model and the tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-2-7b-hf"

# Define phrases and the word in them for looking up embeddings
examples = [
  {
    "phrases": ["She turned on the light in the room.", "She prefers to eat a light meal in the evening."],
    "word": "light"
  },
  {
    "phrases": ["The dog began to bark loudly at the stranger.", "The tree’s bark was rough and covered in moss."],
    "word": "bark"
  },
  {
    "phrases": ["He deposited his paycheck at the bank.", "They had a picnic by the river bank."],
    "word": "bank"
  },
  {
    "phrases": ["She decided to run five miles this morning.", "He will run in the upcoming marathon."],
    "word": "run"
  },
  {
    "phrases": ["He is a strong swimmer and can cross the river easily.", "Her argument was strong and convinced everyone in the room."],
    "word": "strong"
  },
  {
    "phrases": ["The weather outside is very cold today.", "She caught a cold and has been sneezing all day."],
    "word": "cold"
  }
]

# Initialize the Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaModel.from_pretrained(model_dir)

for i, example in enumerate(examples, 1):
    print(f"\nExample {i}: Word '{example['word']}'")
    print("-" * 50)

    print(f"Phrase 1: '{example['phrases'][0]}'")
    print(f"Phrase 2: '{example['phrases'][1]}'")

    # Get embeddings for the word in both phrases
    embeddings = [get_word_embedding_in_phrase(model, tokenizer, phrase, example['word']) for phrase in example['phrases']]

    # Compare the embeddings
    input_result = compare_embedding(embeddings[0]["input_embedding"], embeddings[1]["input_embedding"])
    output_result = compare_embedding(embeddings[0]["contextual_embedding"], embeddings[1]["contextual_embedding"])

    # Print results
    print(f"Input embedding: Are the embeddings exactly the same? {input_result['are_equal']}")
    print(f"Input embedding: Cosine similarity between embeddings: {input_result['similarity']:.4f}")
    print(f"Contextual embedding: Are the embeddings exactly the same? {output_result['are_equal']}")
    print(f"Contextual embedding: Cosine similarity between embeddings: {output_result['similarity']:.4f}")
