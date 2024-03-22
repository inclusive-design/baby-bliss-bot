import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import textstat

# Sentence to convert
sentence_orig = "I had the pleasure of watching a captivating movie that thoroughly engaged my senses and emotions, providing a delightful escape into the realm of cinematic storytelling."
# Expected sentence
sentence_expected = "past:I have pleasure of watching movie captivating that engage thoroughly my senses and emotions providing escape delightful into realm of storytelling cinematic."
# Model-generated sentence
sentence_generated = "past:I watch pleasure of movie captivating that engage thoroughly my senses emotions provide escape into realm storytelling cinematic."

print(f"Original Sentence: \n{sentence_orig}")
print(f"Expected Sentence: \n{sentence_expected}")
print(f"Generated Sentence: \n{sentence_generated}")

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Semantic Coherence
# Compute embedding for both lists
embeddings1 = model.encode(sentence_generated, convert_to_tensor=True)
embeddings2 = model.encode(sentence_expected, convert_to_tensor=True)

# Compute cosine similarities
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

print("Cosine similarity between expected and generated sentences:", cosine_scores.item())

# 2. Novelty and Creativity
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([sentence_orig] + [sentence_generated])

# Calculate similarity with the corpus
similarity = (X * X.T).A[-1][:-1]

# Measure of novelty (1 - mean similarity)
novelty_score = 1 - np.mean(similarity)

print("Novelty score by comparing with the original sentence:", novelty_score)

# 3. Fluency and Readability
readability_score = textstat.flesch_reading_ease(sentence_generated)

print("Readability Score:", readability_score)
