# python compare_synonyms_in_same_sentence.py

import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from utility_funcs import compare_embedding, get_word_embedding_in_phrase
from itertools import combinations

data = [
    {
        "synonyms": ["place", "area", "location", "space"],
        "sentences": [
            "We need to find a quiet place to sit and discuss the project.",
            "This place is perfect for hosting the event; it has everything we need.",
            "He marked the place on the map where we should meet later.",
            "The park has a designated place for picnics and outdoor activities."
        ]
    },
    {
        "synonyms": ["jug", "kettle", "pot"],
        "sentences": [
            "She poured water from the jug into the cups.",
            "After boiling, she carefully carried the jug to the table and began filling everyone's glasses.",
            "He noticed the jug was almost empty, so he went to refill it before continuing with the meal.",
            "The chef used the jug to simmer the broth slowly, ensuring the flavors were perfectly blended before serving."
        ]
    },
    {
        # Contextual similarity scores:
        # ["pitcher", "jug"]: 0.6017
        # ["pitcher", "pot"]: 0.7454
        "synonyms": ["pitcher", "jug", "kettle", "pot"],
        "sentences": [
            "They filled the pitcher with lemonade for the picnic.",
            "The server brought a large pitcher of water to the table.",
            "After a long hike, they sat around the fire, heating the stew in a pitcher.",
            "The chef carefully transferred the sauce to a pitcher, making it easier to pour over the dishes for presentation."
        ]
    },
    {
        # Contextual similarity scores:
        # ["design", "system"]: 0.682
        # ["method", "system"]: 0.6629
        "synonyms": ["plan", "design", "method", "system"],
        "sentences": [
            "The architect presented a detailed plan for the new building that impressed the entire team.",
            "They implemented a new plan to streamline communication between departments",
            "The engineer developed a novel plan for optimizing energy use in the facility.",
            "Before starting the project, they outlined a comprehensive plan to ensure everything went smoothly."
        ]
    }
]

if len(sys.argv) != 2:
    print("Usage: python compare_synonyms_in_same_sentence.py")
    sys.exit(1)

# Load the local Llama model
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Iterate over the dataset to calculate contextual embeddings
results = {}

for entry in data:
    synonyms = entry["synonyms"]
    target_word = synonyms[0]  # The first word is the word used in the sentences

    for sentence in entry["sentences"]:
        # Get the contextual embedding of the target word in the original sentence
        original_embedding = get_word_embedding_in_phrase(model, tokenizer, sentence, target_word)
        embeddings = {target_word: original_embedding["contextual_embedding"]}

        # Replace the target word with each synonym and calculate their embeddings
        for synonym in synonyms[1:]:
            new_sentence = sentence.replace(target_word, synonym)
            synonym_embedding = get_word_embedding_in_phrase(model, tokenizer, new_sentence, synonym)
            embeddings[synonym] = synonym_embedding["contextual_embedding"]

        # Calculate cosine similarity between the original embedding and each synonym's embedding
        similarities = {}

        # Compare all pairs of embeddings
        for (synonym1, embedding1), (synonym2, embedding2) in combinations(embeddings.items(), 2):
            similarity = compare_embedding(embedding1, embedding2)["similarity"]
            pair = f"{synonym1} vs {synonym2}"
            similarities[pair] = similarity

        # Store results for this sentence
        results[sentence] = similarities

# Output the results
for sentence, similarity in results.items():
    print(f"Sentence: {sentence}")
    for synonym, sim_value in similarity.items():
        print(f"  Cosine similarity with '{synonym}': {sim_value:.4f}")
    print()
