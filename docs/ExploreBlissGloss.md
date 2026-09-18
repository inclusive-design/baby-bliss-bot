# Bliss Gloss Exploration

The English Bliss dictionary provides information about glosses, composing Bliss characters, and explanations
for each Bliss symbol, describing the meaning of each symbol. This exploration investigates whether a language
model’s existing English knowledge can be leveraged to help it understand Bliss symbols by using the Bliss
gloss information.

These exploration is performed with [Llama 3.2 8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

## Setup the Environment

1. Download [Llama 3.2 8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to a local
directory;
2. Edit the script to be run by adjusting the `model_dir` variable to point to the local directory where the
model sits;
3. Install the extra python dependencies by running these commands in the project home directory:
```
cd jobs/bliss-gloss 
pip install -r requirements.txt
```

## Understanding Embeddings

The exploration begins with experiments involving three types of embeddings: input, output, and contextual
embeddings for each token.

These experiments reveal several insights:

1. **Embedding Behavior**: 
   - The input and output embeddings of a token are static, while the contextual embedding changes based on context.
   See the [results](../jobs/bliss-gloss/data/cosine_polysemous_embedding.txt) and the [comparison script](../jobs/bliss-gloss/compare_polysemous_embedding.py).
   
2. **Blissymbolics Glosses**:
   - Blissymbolics contains 6,919 symbols. Of these,
   [3,943 symbols](../jobs/bliss-gloss/data/bliss_ids_with_single_token_gloss.json) have single-token glosses,
   while [2,973 symbols](../jobs/bliss-gloss/data/bliss_ids_without_single_token_gloss.json) do not.

3. **Mean Embedding for Multiple Glosses**:
   - According to Blissymbolics experts, multiple glosses for a single Bliss symbol are typically synonymous.
   When a symbol has multiple single-token glosses, one approach to capture its meaning is by calculating the mean
   embedding of all glosses. We assessed the similarity between this mean embedding and each gloss’s input/output
   embeddings:
      * [Full report of comparisons](../jobs/bliss-gloss/data/mean_all_similarity.json).
      * [Report for symbols with cosine similarity below 50%](../jobs/bliss-gloss/data/mean_low_similarity_0.5_34.json) (34 symbols).
      * [Report for symbols with cosine similarity below 70%](../jobs/bliss-gloss/data/mean_low_similarity_0.7_1023.json) (1,023 symbols).
   - In most cases, the mean embedding is not an adequate representation of all glosses for a symbol.

4. **Direct Gloss Comparison**:
   - [A direct comparison]((../jobs/bliss-gloss/compare_gloss_by_cosine.py)) between glosses was performed for
   each Bliss symbol with multiple single-token glosses. Findings:
      * [This report](../jobs/bliss-gloss/data/consine_mismatch_on_input_output.json) indicates low cosine similarities
      between synonyms' input and output embeddings.
      * However, cosine similarities between synonyms' contextual embeddings are higher. See
      [this report](../jobs/bliss-gloss/data/cosine_mismatch_on_contextual.json) (only 88 out of 1,872 pairs showed
      similarity lower than 80%).

5. **Contextual Embedding in Sentences**:
   - We further evaluated contextual embedding similarity when synonyms are used interchangeably in the same sentence.
   [This report](../jobs/bliss-gloss/data/contextual_similarity_synonyms_in_same_sentences.txt) demonstrates that
   similarity scores are context-dependent, with some low-similarity synonyms showing high similarity in specific contexts.

6. **Experiment with Attention-Weighted Embeddings for Multi-Token Glosses**:
   - In this experiment, attention-weighted embeddings are used to create a single embedding for glosses composed
   of multiple tokens. [This script](../jobs/bliss-gloss/weighted_embedding_for_multi_token_gloss.py) calculates token importance scores based on attention weights for each token in the gloss, then uses these scores to generate weighted
   input and output embeddings. These embeddings are assigned to a new token representing the Bliss symbol.
   - The evaluation process compares prediction probabilities for the original gloss and the newly generated Bliss
   token within the same prefix context. However, results indicate this method may be suboptimal, as there is a big gap
   in prediction ranks between the Bliss token and the first token of the original gloss.

## Cleaning Bliss Gloss

Bliss gloss data requires cleaning and standardization before it can be applied to a language model. We used a script
and manual review for this process:

1. **Initial Cleaning**:
   - [This script](../jobs/bliss-gloss/clean_bliss_gloss.py) removes certain substrings, such as anything following
   `_(")`, and replaces underscores (`_`) with spaces. It generates a JSON file mapping symbol IDs to lists of glosses.
   
2. **Manual Review**:
   - In the generated JSON file, remove glosses from the 34 symbols where embedding similarity is lower than 50%
   when compared with the mean embedding. (Refer to the 
   [low-similarity report](../jobs/bliss-gloss/data/mean_low_similarity_0.5_34.json).)
   
3. **Synonym Similarity Review**:
   - In the same JSON file, review the 88 synonym pairs with contextual embedding similarity below 80%. Remove ones
   that have high semantic differences with other synonyms.

The cleaned gloss data can be found in [this JSON file](../jobs/bliss-gloss/data/bliss_gloss_cleaned_synonyms.json).

## Applying Gloss Embeddings to New Bliss Tokens

To integrate Bliss gloss information into Llama 3.1, we developed the script
[add_bliss_tokens.py](../jobs/bliss-gloss/add_bliss_tokens.py), which:

1. Iterates through Bliss symbols with single-token glosses found in the
[cleaned gloss data](../jobs/bliss-gloss/data/bliss_gloss_cleaned_synonyms.json).
2. Adds a unique token for each symbol in the format `[BLISS_{symbol_id}]`.
3. Assigns input and output embeddings to each token, testing two methods:
   - **Method 1**: Uses the embedding of the first single-token gloss, enabling the new token to behave identically
   to this gloss.
   - **Method 2**: Uses the mean embedding of all single-token glosses for the symbol. (This approach does not
   yield the desired prediction accuracy; see the [prediction comparison](https://docs.google.com/spreadsheets/d/1FrHrHilf3Nsgb-gEsTlBpsY2MUG75VZ56LP99ybcVNw/edit?gid=0#gid=0).)

With English gloss embeddings applied, these special Bliss tokens are treated as English by Llama.

Additionally, by zeroing out the output embeddings of all non-Bliss tokens in the output layer, Llama can respond
solely in Bliss tokens.

## Issues

Here’s a refined version of your text with improved clarity, structure, and flow:

---

## Issues

1. Ambiguity in English Glosses  
English glosses often carry multiple meanings, which creates challenges when mapping them to Blissymbolics.
For example, the word *bark* can refer to either the sound a dog makes or the outer layer of a tree. By comparison, Blissymbolics uses distinct symbols for each meaning, so there is no ambiguity when using a Bliss symbol. When
the embedding for the English word *bark* is assigned to a Bliss token, the Bliss token cannot distinguish which
specific meaning of *bark* corresponds to the intended Bliss symbol.

2. Difficulty Handling Multi-Token Glosses  
Another challenge is when an English gloss consists of multiple tokens. Currently, there is no effective method to
combine the embeddings of multiple tokens into a single embedding that can be assigned to a single Bliss token.
This limitation prevents accurate representation of multi-token glosses within the Blissymbolics.

3. Ambiguity in Blissymbolics Symbols  
A further challenge arises when a single Blissymbolics symbol represents multiple meanings. For example, Bliss ID
24819 can mean either "lowness" or "shortness" in the context of height, while Bliss ID 14916 can refer to "I",
"me", or "myself". This polysemy introduces ambiguity when assigning an embedding to a Bliss token, as the embedding
must account for all possible meanings associated with that symbol.
