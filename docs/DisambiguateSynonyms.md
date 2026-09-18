# Word Disambiguation in Blissymbolics

## Overview

Our current method for integrating Blissymbols with language models such as LLaMA
uses the English gloss of each symbol as an intermediary. A gloss is one or more
English words that approximate the meaning of a Blissymbol. However, this introduces
ambiguity: while Blissymbols are typically semantically precise, English words often
have multiple meanings.

## Challenges

This gloss-based approach presents two primary challenges, depending on the gloss
structure:

1. **Ambiguous Single-token gloss (e.g., "bark")**
   The English word *bark* can refer to the sound made by a dog or the outer layer
   of a tree. Blissymbolics, by contrast, represent each meaning with a distinct symbol.

   When initializing a new token for a Blissymbol, we can reuse the input embedding
   of the existing English token (e.g., "bark"). Contextual disambiguation occurs
   in the model’s hidden layers. However, a unique output embedding must still be
   computed to reflect the specific Blissymbol meaning.

2. **Multple synonym glosses (e.g., "break, fracture, injure, damage")**
   Some Blissymbols, such as the one with BCI-AV ID 12869, are glossed with multiple
   English synonyms: "break, crack, fracture, tear, injure, damage." While these words
   share a core meaning, each also carries distinct contextual nuances. This is a
   common pattern across many Blissymbols.

   In such cases, both input and output embeddings must be computed to capture the
   shared semantic space of the gloss terms.

This document explores solutions for these two challenges.

## Computing the Input Embedding

When a Blissymbol is associated with multiple synonym glosses, a new input embedding
must be computed to represent the shared semantic space of those synonyms. We
experimented with three approaches:

1. **Averaging Input Embeddings**
2. **Principle Component Analysis**
3. **Attention-based Aggregation**

### Approach 1: Averaging Input Embeddings

This method computes the element-wise average of the embedding vectors for all synonyms
in the cluster. Despite its simplicity, this approach yielded strong results in text
generation tasks, making it a practical baseline.

### Approach 2: Principle Component Analysis (PCA)

PCA identifies the directions (principal components, or PCs) along which the synonym
embeddings vary the most. By projecting the average embedding onto a subspace defined
by the top PCs, we aim to capture the core shared meaning while reducing noise.

**Procedure:**

1. Stack Embeddings: Treat the synonym embeddings as points in a high-dimensional space.
2. Calculate Covariance Matrix: Measure how the embeddings vary together.
3. Extract Principal Components: Find the eigenvectors of this matrix. The eigenvector
with the largest eigenvalue is PC1, and so on.
4. Project onto Subspace: Select the top N PCs that capture the most variance. Project
the average embedding onto this subspace.

**Choosing the Number of PCs (N):**

Two methods were tested:

1. Use Explained Variance Threshold
2. Use Kneed Library
3. Use Primary Principal Component

#### Explained Variance Threshold

We tested thresholds of 0.95 and 0.99, meaning 95% and 99% of the total variance is
retained. In both cases, PCA returned 3 principal components. However, the resulting
embeddings did not yield satisfactory results in text generation.

#### Kneed Library

* With a sensitivity of 1 and no PC limit, 4096 components were returned—equal to the
dimensionality of LLaMA embeddings—resulting in an embedding identical to the average.
* Since only 4 synonym embeddings were used ("break", "fracture", "injure", "damage"),
the maximum dimensionality of the subspace is 3. Restricting the output to 3 PCs
resulted in only 1 PC being selected, which also produced suboptimal results.

#### Primary Principal Component

* Use the primary principal component (PC1) as the initial input embedding and output
embedding.

### Approach 3: Attention-based Aggregation

Instead of a fixed weight (like in weighted averaging), attention dynamically calculates
an "importance score" for each synonym within the context of the entire synonym cluster.
Synonyms that are more semantically central to the cluster will receive higher scores.

**Procedure:**

1. Define a "Query" Vector: Use the average of all synonym embeddings as the query
vector `q`, representing the overall meaning.
2. Calculate Similarity Scores: For each synonym embedding s_i, calculate a similarity
score with the query vector q using the dot product: score_i = q · s_i. A high score
means the synonym s_i is very similar to the overall cluster average.
3. Normalize Scores (Softmax): Apply a softmax function to all the scores. This converts
them into a probability distribution, where all scores are positive and sum to 1. For
the attention weights (α_i): α_i = softmax(score_i)
4. Compute Weighted Sum: pute the final embedding as a weighted sum:
[MEANING_SHINY]_embedding = Σ (α_i * s_i)

In practice, this method produced an embedding nearly identical to the average embedding.

### Script and Test Results

We evaluated all three methods using the Blissymbol with BCI-AV ID 24852, which has four
single-token glosses: break, fracture, injury, and damage. The goal was to add a new
token [BLISS_24852] to the LLaMA model, compute its input and output embeddings using
each method, and evaluate performance in word prediction and text generation tasks.

- **Script**: [input_embedding_PCA_and_self-attention.py](../jobs/bliss_gloss/disambiguation/input_embedding_PCA_and_self-attention.py)

- **Test Results**

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th colspan="4" style="text-align:center;">Input Embedding</th>
      <th colspan="3" style="text-align:center;">Output Embedding</th>
      <th style="text-align:center;">Test Result Log</th>
    </tr>
    <tr>
      <th style="text-align:center;">Calculation Method</th>
      <th style="text-align:center;">Cosine Similarity<br>Btw Target and Average</th>
      <th style="text-align:center;">Euclidean Distance<br>Btw Target and Average</th>
      <th style="text-align:center;">Calculation Method</th>
      <th style="text-align:center;">Cosine Similarity<br>Btw Target and Calculated</th>
      <th style="text-align:center;">Euclidean Distance<br>Btw Target and Calculated</th>
      <th style="text-align:center;">Quality of Prediction Result<br>(Test output embedding)</th>
      <th style="text-align:center;">Quality of Text Generation Result<br>(Test input embedding)</th>
      <th style="text-align:center;">Test Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Average</td>
      <td>1</td>
      <td>0</td>
      <td>Calculated on positive &amp; negative contexts</td>
      <td>1</td>
      <td>0</td>
      <td>Best</td>
      <td>Good</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_average_calculated.log)</td>
    </tr>
    <tr>
      <td>PCA (threshold: 0.95)<br>3 principle components found</td>
      <td>0.1948</td>
      <td>0.3766</td>
      <td>PCA (threshold: 0.95)<br>3 principle components found</td>
      <td>0.0467</td>
      <td>1.81</td>
      <td>Bad</td>
      <td>Good except last two generations</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_PCA_threshold_0.95.log)</td>
    </tr>
    <tr>
      <td>PCA (threshold: 0.99)<br>3 principle components found</td>
      <td>0.1948</td>
      <td>0.3766</td>
      <td>Calculated on positive &amp; negative contexts</td>
      <td>1</td>
      <td>0</td>
      <td>Good</td>
      <td>Bad</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_PCA_threshold_0.99.log)</td>
    </tr>
    <tr>
      <td>PCA (Kneed library)<br>Sensitivity: 1<br>no PC restrictions<br>4096 principle components found</td>
      <td>1</td>
      <td>0</td>
      <td>PCA (Kneed library)<br>4096 principle components found</td>
      <td>0.0652</td>
      <td>1.8471</td>
      <td>Good</td>
      <td>Good</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_PCA_kneed.log)</td>
    </tr>
    <tr>
      <td>PCA (Kneed library)<br>Sensitivity: 1<br>restricted to 3 PCs<br>1 principle components found</td>
      <td>0.0633</td>
      <td>0.0708</td>
      <td>Calculated on positive &amp; negative contexts</td>
      <td>1</td>
      <td>0</td>
      <td>Good</td>
      <td>Bad</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_PCA_kneed_restricted_to_N-1.log)</td>
    </tr>
    <tr>
      <td>PCA (Primary principal component)</td>
      <td>-0.0633</td>
      <td>1.0936</td>
      <td>PCA (Primary principal component)</td>
      <td>-0.0175</td>
      <td>1.124</td>
      <td>Bad</td>
      <td>Good</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_OE_PC1.log)</td>
    </tr>
    <tr>
      <td>Attention based</td>
      <td>0.9999</td>
      <td>0.0064</td>
      <td>Attention based</td>
      <td>0.066</td>
      <td>1.847</td>
      <td>Good</td>
      <td>Good</td>
      <td>[Test Result](../jobs/bliss_gloss/disambiguation/test_results/IE_self_attention.log)</td>
    </tr>
  </tbody>
</table>

## Computing the Output Embedding

We use a data-driven approach to compute the output embedding for the new token by
leveraging context sentences.
Two sets of sentences are created:

**1. Positive Context Sentences**
These sentences strongly predict "bark" in the sense of an animal sound. For each
sentence, we extract:
- The **contextual embedding** (hidden states from the last hidden layer) of the
last token.
- The **logits** for the token "bark"

**2. Negative Context Sentences**
These sentences either predict "bark" in the sense of tree bark or do not predict
"bark" at all. For these sentences, we extract:
- The **contextual embedding** of the last token.
- The **logits** for "bark," assigning a negative value to ensure the new token
does not receive a high probability in these contexts.

**Mathematical Formulation**
For each context sentence, the calculation of the logits follows this formula:
```
logit_c = dot_product(contextual_embedding_c, output_embedding_of_new_token)
```

Given:
- **N**: Total number of context sentences
- **D**: Model embedding dimension
- **OE**: Output embedding of the new token

We aggregate extracted data into:
1. **T_CE**: A tensor of shape *(N, D)* holding all contextual embeddings.
2. **T_L**: A tensor of shape *(N, 1)* holding all logits.

The matrix equation to calculate logits for all context sentences in one shot is:
```
T_CE * OE = T_L
```

Two approaches are experimented:

1. **Optimization Approach**
2. **Inversion of Matrix Multiplication Approach**

### Approach 1: Optimization

This approach starts with a random output embedding and optimizes it to minimize
the difference between expected and actual logits.

Three versions of this approach were tested. See the comments in
[optimize_output_embedding.py](../jobs/bliss_gloss/disambiguation/optimize_output_embedding.py)
for details on each version.

#### Test Results
The optimization approach produced inconsistent results. The ranking of the new
token was good in some contexts but poor in others. Test results are available in:

- [OE-optimization-V1-result_1000_0.01_0.2.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_1000_0.01_0.2.log)
- [OE-optimization-V1-result_500_0.01_0.3.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_500_0.01_0.3.log)
- [OE-optimization-V1-result_800_0.01_0.1.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_800_0.01_0.1.log)
- [OE-optimization-V2-result_700_0.01_0.2.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V2-result_700_0.01_0.2.log)
- [OE-optimization-V2-result_800_0.001_0.3.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V2-result_800_0.001_0.3.log)
- [OE-optimization-V3-result_800_0.01.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V3-result_800_0.01.log)

### Approach 2: Inversion of Matrix Multiplication

This approach inverts the matrix equation directly using
[torch.linalg.lstsq()](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).

- **Script**: [calc_output_embedding.py](../jobs/bliss_gloss/disambiguation/calc_output_embedding.py)

#### Test Results
This method produced significantly better results across all test sentences. The
output embedding calculated with this method led to more accurate predictions. The
test result is available in:

- [OE-invert-matrix-multiplication.log](../jobs/bliss_gloss/disambiguation/test_results/OE-invert-matrix-multiplication.log)

### Supporting Data and Scripts

#### Dataset
- **Script**: [dataset_animal_bark.py](../jobs/bliss_gloss/disambiguation/data/dataset_animal_bark.py)

This script defines three arrays that contain context sentences for training
or testing:
1. Positive context sentences: The model assigns a high probability to "bark"
in the sense of animal bark as a top prediction.
2. Negative context sentences: The model assigns a low probability to "bark"
in the sense of animal bark in these contexts.
3. Testing context sentences: Mixed-context examples for testing

#### Sentence Prediction Evaluation
Checks the model’s prediction of "bark" across all training and testing sentences.
- **Script**: [sentence_predictions.py](../jobs/bliss_gloss/disambiguation/sentence_predictions.py)
- **Results**: [dataset_animal_bark_predictions.log](../jobs/bliss_gloss/disambiguation/test_results/dataset_animal_bark_predictions.log)

#### Output Embedding Comparison
Compares cosine similarity between embeddings generated by the two approaches.
The optimization approach uses version 3 with:
- **Epochs**: 200
- **Learning Rate**: 0.01

- **Script**: [output_embedding_compare.py](../jobs/bliss_gloss/disambiguation/output_embedding_compare.py)
- **Results**: [output_embedding_compare.log](../jobs/bliss_gloss/disambiguation/test_results/output_embedding_compare.log)

### Conclusion

The **inversion of matrix multiplication** approach provides the best results
for computing output embeddings of disambiguated Blissymbol tokens. The optimization
approach, while feasible, produces inconsistent outcomes.Future work will explore
refining input embeddings for multi-token glosses.

### Conclusion

1. **PCA with explained variance thresholds** (e.g., 0.95 or 0.99) did not yield
high-quality input embeddings and underperformed in evaluation tasks.
2. **PCA with the Kneed method** and **attention-based aggregation** both produced
embeddings that were either identical or nearly identical to the simple average
embedding. In these cases, averaging is more efficient and equally effective.
3. **PCA remains a promising direction** for capturing nuanced semantic structure,
but it requires further experimentation, especially with larger synonym sets, to
demonstrate consistent benefits.
4. Fow now, **primary principal component** and **averaged input embedding** are the
most effective and efficient strategy for initializing the input embedding.

# Final Conclusion
1. **Recommended approach**: the most effective and efficient strategy is to use:
* **primary principal component** and **Averaged input embedding** as the initial input
embedding.
* **Calculated output embedding** as the initial output embedding.
