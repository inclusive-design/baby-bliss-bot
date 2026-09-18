# Add Grammatical Indicators

This document describes three methods explored for integrating Blissymbolics grammatical indicator symbols into a Llama model.

The initial focus is on the *action indicator* (BCI-AV-ID 8993) in Blissymbolics. This indicator, when used with a noun symbol, converts the meaning into the infinitive verb form of that noun. For example, when combined with the symbol for "fire", it produces the meaning "to burn".

The goal is to introduce a new token, `[BLISS_8933]`, into the Llama model such that it performs this grammatical transformation.

## Methods

Three methods are investigated. They differ in:
1. the template used to combine `[BLISS_8933]` with a noun
2. the initialization of the `[BLISS_8933]` token:

### Method 1: "To"-Embedding Initialization

* **Template:** `[[BLISS_NOUN];[BLISS_8933]]`  
* **Embedding Initialization:** Use the input and output embeddings of the token `" to"`.

### Method 2: Difference Vector Averaging

* **Template:** `[[BLISS_NOUN];[BLISS_8933]]`
* **Embedding Initialization:** Calculate `[BLISS_8933]` embedding as the mean vector of several noun-to-infinitive differences:  
   ```python
   mean([
       embedding("to expand") - embedding("expansion"),
       embedding("to decide") - embedding("decision"),
       embedding("to remove") - embedding("removal"),
       ...
   ])
   ```

### Method 3: Infinitive Isolation Embedding

* **Template:** `[BLISS_8933][BLISS_NOUN]`
* **Embedding Initialization:**
   - Input embedding: copy from `" to"`  
   - Output embedding: compute a "pure" infinitive representation by isolating the grammatical meaning of `" to"` (excluding its other uses).

After initializing `[BLISS_8933]`, QLoRA fine-tuning is performed to refine its embedding and attention layers. The effectiveness of the token is evaluated through prediction and text generation tests to determine whether the grammatical transformation is successfully learned.

### Method 4: Random Initial Embedding

Method 4 uses the same template and fine-tuning method as method 3, except the input and output embeddings are initialized
with ramdom embeddings to reduce the bias towards the meaning of "to".

## Test Results

* [Method 1 test result](../jobs/bliss-gloss/integrate_grammar_indicators/logs/add_8993_init_bothAsTo.log)
* [Method 2 test result](../jobs/bliss-gloss/integrate_grammar_indicators/logs/add_8993_init_bothAsDiff.log)
* [Method 3 test result](../jobs/bliss-gloss/integrate_grammar_indicators/logs/add_8993_init_inputAsTo_outputCalc.log)
* [Method 4 test result](../jobs/bliss-gloss/integrate_grammar_indicators/logs/add_8993_init_bothAsRandom.log)

None of the methods produced satisfactory results. The grammatical transformation was not learned effectively. Observed issues include:

1. **Flat training curves:** Loss remained mostly unchanged throughout fine-tuning, both for training and evaluation.
2. **Minimal embedding change:**  
   - For the "difference" method, cosine similarity before and after fine-tuning was nearly identical, and Euclidean distance changed only slightly.  
   - For the `"to"`-based methods, cosine similarity changed slightly, while Euclidean distance remained unchanged.
3. **Ineffective text generation:** In `"to"`-based methods, `[BLISS_8933]` continued to carry the original meaning of `"to"`, and the noun retained its noun meaning. No grammatical shift to the infinitive verb form was observed in all methods.

## Technical Steps

1. Run [`create_dataset_action_indicator.py`](../jobs/bliss-gloss/integrate_grammar_indicators/create_dataset_action_indicator.py) to generate [prompt templates](../jobs/bliss-gloss/integrate_grammar_indicators/output/action_indicator_prompts.txt) for dataset creation. 
* **Note**: This script also generates [a JSON file](../jobs/bliss-gloss/integrate_grammar_indicators/data/action_indicator_id_pairs.json) listing noun–verb pairs in which the action indicator triggers the grammatical transformation.
2. Use a large language model (e.g., ChatGPT) to generate English training examples from the prompts. Save the results as [`dataset_action_indicator_eng.py`](../jobs/bliss-gloss/integrate_grammar_indicators/data/dataset_action_indicator_eng.py).
3. Run [`convert_dataset_eng_to_bliss.py`](../jobs/bliss-gloss/integrate_grammar_indicators/convert_dataset_eng_to_bliss.py) to convert the English dataset into Bliss token format, applying the appropriate method-specific template. The output is [`dataset_action_indicator_bliss.py`](../jobs/bliss-gloss/integrate_grammar_indicators/data/dataset_action_indicator_bliss.py).
4. Create additional test data for evaluating both prediction and text generation performance of `[BLISS_8933]`.
5. Combine the training and testing data created at step 3 & 4 into a single dataset file: [`initial_8933.py`](../jobs/bliss-gloss/integrate_grammar_indicators/data/initial_8933.py).
6. For Method 3, create an additional dataset: [`dataset_to_as_infinitive_marker.py`](../jobs/bliss-gloss/integrate_grammar_indicators/data/dataset_to_as_infinitive_marker.py). This is used to calculate the output embedding representing the infinitive-only meaning of `"to"`.
7. Run the main test script [`add_action_indicator_8933.py`](../jobs/bliss-gloss/integrate_grammar_indicators/add_action_indicator_8933.py) to evaluate each method.
