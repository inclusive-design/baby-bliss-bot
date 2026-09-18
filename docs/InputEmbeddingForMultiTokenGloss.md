# Calculate Input Embedding for Multi-Token Glosses

Bliss symbols often have glosses composed of multiple tokens. When introducing a new token to represent such a symbol, it's common to use its multi-token gloss as a semantic bridge to integrate it into a language model. One critical step in this process is initializing the **input embedding** for the new token effectively, as it influences the success of subsequent fine-tuning.

This document outlines and evaluates different approaches for initializing the input embedding and their impact on fine-tuning and final model performance.

---

## Overview

- **Goal:** Introduce a new token with a multi-token gloss (e.g., `"wool shop"`) and determine the most effective way to initialize its input embedding.

## Method 1: Input Embedding Optimization

The goal of the optimization method to calculate a single input embedding that semantically represents a multi-token phrase without altering the model's underlying weights. Two ways of optimization are tested:
1. Optimize a standalone input embedding in a variable
2. Optimize the input embedding of a single, newly added token in a language model

The optimization objective is to minimize the difference between two contextual embeddings:
1. the target hidden state produced by the language model after processing a prefix sentence followed
by the original phrase
2. the hidden state produced after processing the same prefix followed by the target input embedding.

The process involves these steps:
1. Data Preparation: Create a dataset of training sentences, validation sentences, and testing prompts.
2. Semantic Initialization: The initial input embedding is initialized with the average of the static embeddings
of the tokens in the phrase, providing a good starting point.
3. Optimization: With the base model's parameters frozen, an Adam optimizer iteratively refines the input embedding to minimize the Cosine Embedding Loss between its output and the target hidden states. The best-performing embedding is selected based on the lowest validation loss.
4. Evaluation: The final, optimized input embedding and a calcualted output embedding is assigned to a new token in the model's vocabulary and is evaluated on its ability to perform next-word prediction and coherent text generation.

Note: See [Disambiguate Synonyms documentation](./docs/DisambiguateSynonyms.md) regarding how the output embedding is calculated.

### Test Results

**Condition**:
learning rate = 0.0005
epochs = 100
initial input embedding = average of input embedding of composing tokens

| Method | Change of Validation Loss | Change of Cosine Similarity | Change of Euclidean Distance | Optimization Time	| Test Result |
|--------|---------------------------|-----------------------------|------------------------------|----------------------|--------------|
| [Optimize a variable](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization_variable.py) | 0.083170 -> 0.021390 at epoch 87 | 0.3186 | 1.0985 | 6 mins | [Test Result](../jobs/bliss-gloss/multi-tokens-phrase/test_results/29111_wool_shop/input_embedding_optimization_variable_0.0005_100.log) |
| [Optimize a new model token](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization_new_model_token.py) | 0.083170 -> 0.021390 at epoch 87 | 0.3186 | 1.1064 | 5 mins | [Test Result](../jobs/bliss-gloss/multi-tokens-phrase/test_results/29111_wool_shop/input_embedding_optimization_new_model_token_0.0005_100.log) |

### Conclusion

1. Optimization is effective in improving the single input embedding representing multiple-token phrase.
2. Optimizing a model token or optimizing an independent variable produces similar result.

## Method 2: With a follow up fine-tuning
- **Process:** 
  1. Initialize both input and output embeddings.
  2. Fine-tune the model using a dataset that contains the new token.
  3. Evaluate the model's understanding and generation capabilities with the new token.

---

### Step 1: Initialize Input and Output Embeddings

We use a consistent method to compute the **output embedding** based on previous work regarding [Disambiguate Synonyms](./docs/DisambiguateSynonyms.md). This document focuses on different strategies for initializing the **input embedding**.

#### Input Embedding Initialization Strategies:

1. **Optimized input embedding with a ramdom initial value** (via contextual optimization)
2. **Averaged input embedding** of all composing tokens
3. **Input embedding of the last token** in the gloss
4. **Random input embedding**
5. **Random input and output embeddings**

---

Since Strategies 2–5 involve straightforward initialization techniques, we focus here on **Strategy 1**, which uses contextual optimization to compute the input embedding:

#### Strategy 1: Optimized Input Embedding via Contextual Optimization

- **Example Gloss:** `"wool shop"`

- **Overview:**  
  This method generates an optimized input embedding for the new token by aligning it with the contextual behavior of its gloss in natural sentences. The initial input embedding is a random embedding.

- **Steps:**
  1. **Prepare Context Sentences**
     - Create two sets of partial sentences:
       - **Positive Contexts**: Sentences where `"wool shop"` is a highly likely next phrase.
       - **Negative Contexts**: Sentences where `"wool shop"` is an unlikely continuation.
     - *Examples can be found in the variables* `training_positive_context_sentences` *and* `training_negative_context_sentences` *in* [`dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/dataset_29111_wool_shop.py).

  2. **Run Optimization**
     - Use the script [`input_embedding_optimization.py`](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization.py) to:
       - Compute the **target contextual embeddings** of `"wool shop"` across the context sentences.
       - **Iteratively optimize** a randomly initialized input embedding so that it matches the averaged contextual behavior of the gloss.

- **Result:**  
  This approach ensures the new token embedding reflects the semantic context in which the gloss commonly appears, improving its initialization for downstream fine-tuning.

---

### Step 2: Fine-Tuning Procedure

Once the input and output embeddings are initialized, fine-tuning adjusts:
- The new token's input and output embeddings
- Attention layers in the transformer

This step uses the script [`qlora_embedding_exploration.py`](../jobs/bliss-gloss/multi-tokens-phrase/qlora_embedding_exploration.py) to perform a QLoRA fine-tuning.

*The Example of the fine-tuning dataset can be found in the variables* `fine_tuning_sentences` *in* [`dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/dataset_29111_wool_shop.py).

---

### Test Results

| Input Embedding Initialization | Best Epoch | Eval Loss | Train Loss | Cosine Similarity (Before/After Fine-Tuning) | Test Quality | Time Per Epoch | Test Result |
|-------------------------------|------------|-----------|------------|----------------------------------------------|--------------|------------|------------|
| **Optimized input; calculated output** | 6 | 0.3367 | 0.1824 | Input: 1.000 | Good | ~0.87 min | [Test Result](./test_results/29111_wool_shop/qlora_fine_tune_embedding_no_reset_0.0003_10_10.log) |
| **Averaged input; calculated output** | 6 | **0.3336** | **0.1816** | Input: 0.0344 (vs random), 1 (vs optimized) | **Best** | ~0.84 min | [Test Result](./test_results/29111_wool_shop/qlora_fine_tune_average_only_attention_layer_0.0003_10_10.log) |
| **Last token input ("shop"); calculated output** | 6.33 | 0.3371 | 0.2011 | Input: 0.0533 (vs random), 1 (vs optimized) | Good | ~0.84 min | [Test Result](./test_results/29111_wool_shop/use_contextual_as_input_embedding.log) |
| **Random input; calculated output** | 5.33 | 0.3789 | 0.2846 | Input: -0.0389, Output: 1.000 | Mostly Good, one anomaly | ~0.87 min | [Test Result](./test_results/29111_wool_shop/qlora_fine_tune_embedding_random_input_0.0003_10_10.log) |
| **Random input & output** | 16.33 | 0.8693 | 0.2347 | Input: 0.0370, Output: 0.0081 (vs random) | Inconsistent; worse generation | ~0.87 min | [Test Result](./test_results/29111_wool_shop/qlora_fine_tune_random_both_attention_layer_with_new_token_embedding_0.0003_30_10.log) |

---

### Conclusion

The **most effective approach** is:

> **Initialize the input embedding using the averaged embeddings of all composing tokens** in the gloss, combined with the calculated output embedding.

### Why this works best:
- Lowest **evaluation and training loss**
- High **semantic alignment** with optimized embedding
- Best performance in **generation** and **prediction tasks**
- Faster convergence with fewer fine-tuning epochs

---

## 📎 References

- [Disambiguate Synonyms](./docs/DisambiguateSynonyms.md)
- [Optimization Script](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization.py)
- [QLoRA fine-tuning Script](../jobs/bliss-gloss/multi-tokens-phrase/qlora_embedding_exploration.py)
- [Context Dataset: `dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/)
