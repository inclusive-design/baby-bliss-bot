# Adding a New Token for a Bliss Symbol in LLaMA

This documentation outlines the steps to add a new token for a Bliss symbol into the LLaMA model. As an example, we will use the Bliss symbol with ID 24819, which represents "lowness" or "shortness" in the context of height.

The working directory for all steps is: [`jobs/bliss-gloss/multi-tokens-phrase/`](../jobs/bliss-gloss/multi-tokens-phrase/).

## Step 1: Create and Prepare the Dataset

### Step 1.1 Generate Synthetic Datasets Using LLMs

You can use a large language model like ChatGPT to generate synthetic data. Refer to prompts.md for sample prompts used to create various datasets.

Each dataset should be diverse and span a wide range of contexts and grammatical structures. All datasets are Python lists.

The datasets required are:
1. **Positive Partial Context Sentences****

Sentences that provide strong contextual cues leading to the prediction of "lowness" or "shortness" in the context of height.

2. **Negative Partial Context Sentences**

Sentences that should not lead to the prediction of "lowness" or "shortness" in the height context. These may include:
* Irrelevant contexts where those words might appear
* Contexts leading to different word predictions

3. **Fine-Tuning Sentences**

Sentences that clearly use "lowness", "shortness", or both in the height context. These will be used to fine-tune the model on the new token.

4. **Text Generation Prompts for Testing**

Partial sentences that contain the placeholder {placeholder} where the new token or gloss word should appear. These are used to evaluate how well the fine-tuned model understands the connection between the new token and its context.

### Handling Rare or Ambiguous Glosses

Some glosses may be infrequent or difficult to incorporate naturally into partial context sentences. In such cases, you can prompt the large language model (LLM) to generate full sentences containing the target gloss. Then use the helper script below to truncate those sentences from the point where the gloss appears, producing more focused partial inputs for your dataset.

* Script: [`jobs/bliss-gloss/multi-tokens-phrase/generate_partial_context_sentences.py`](jobs/bliss-gloss/multi-tokens-phrase/generate_partial_context_sentences.py)

    This script reads the training_positive_context_sentences and training_negative_context_sentences variables from the specified dataset file. These should be Python lists of strings—each string being a full or partial sentence that contains one or more of the specified glosses.

    The script performs the following:

    * Locates the last occurrence of any specified gloss in each sentence.
    * Truncates the sentence from that gloss onward (inclusive).
    * If no gloss is found in a sentence, it remains unchanged.

* Usage:

```bash
python generate_partial_context_sentences.py {file_path} {gloss_list}
```

* Example:

```bash
python generate_partial_context_sentences.py "./data/dataset_24918_lowness_shortness.py" '[" lowness", " shortness"]'
```

* Output:

    The script appends the truncated sentences as new versions of the same variables—training_positive_context_sentences and training_negative_context_sentences—in the provided dataset file.

    After the script finishes, the file will contain two copies of each variable:

    * The original list (pre-truncation)
    * The newly truncated list (appended)

    **Note** You will need to manually review the results and remove the original values if the truncated versions are acceptable.

### Step 1.2: Screen Datasets

After generating the datasets, you should screen them to ensure that the glosses behave as expected in their respective contexts:

* **Positive partial sentences** should result in **high ranking** predictions for the target glosses.
* **Negative partial sentences** should result in **low ranking** predictions.

Use the following script to evaluate sentence quality:

* Script: [`jobs/bliss-gloss/multi-tokens-phrase/sentence_predictions.py`](jobs/bliss-gloss/multi-tokens-phrase/sentence_predictions.py)

  This script processes both `training_positive_context_sentences` and `training_negative_context_sentences`.

* Configurable Thresholds

    Adjust the following variables inside the script to control which sentences are accepted:

    * `RANK_POSITIVE_THRESHOLD`: The maximum allowed rank for at least one of the target gloss tokens for the sentence to be considered a valid **positive**.
    * `RANK_NEGATIVE_THRESHOLD`: The maximum allowed rank for **all** target gloss tokens for the sentence to be considered a valid **negative**.

* Example Usage

```bash
python sentence_predictions.py '["lowness", "shortness"]'
```

* Output

    The script iterates through each partial sentence and reports:

    * The rank of the first token of each target gloss.
    * The top 5 predicted tokens for each sentence.

    It then returns new Python lists containing only the sentences that meet the defined thresholds, allowing you to filter out low-quality examples from your dataset.

### Step 1.3: Finalize the Dataset

Once the datasets are validated, split the first three datasets into training and testing subsets. Save them in a file named:

```
dataset_{BlissID}_{gloss1}_{gloss2}_..._{glossN}.py
```

* Note: Remove spaces from gloss names in the filename.
* Place the file in the directory: [`jobs/bliss-gloss/multi-tokens-phrase/data/`](jobs/bliss-gloss/multi-tokens-phrase/data)

The dataset file should define the following Python list variables:

```
training_positive_context_sentences     # 80% of positive partial context sentences
testing_positive_context_sentences      # 20% of positive partial context sentences
training_negative_context_sentences     # 80% of negative partial context sentences
testing_negative_context_sentences      # 20% of negative partial context sentences
fine_tuning_sentences                   # 80% of fine-tuning sentences
validation_sentences                    # 20% of fine-tuning sentences
testing_text_generation_prompts         # 100% of generation prompts
```

## Step 2: Prepare Scripts

### Scripts Involved

1. **Dataset file:** `dataset_{BlissID}_{gloss1}_{gloss2}_..._{glossN}.py`
   * This dataset file should follow the naming convention as described in Step 1.
   * It must define all necessary Python lists used during initialization, fine-tuning, and testing.

2. **Main script:** [`jobs/bliss-gloss/multi-tokens-phrase/qlora_cover_multi_glosses.py`](../jobs/bliss-gloss/multi-tokens-phrase/qlora_cover_multi_glosses.py)
   This script performs the core logic of adding a new token and fine-tuning the model:
   * Computes the **initial output embedding** of the new token based on `training_positive_context_sentences` and `training_negative_context_sentences`.
   * Computes the **initial input embedding** as the average of all token embeddings across the target glosses.
   * Adds a new token to the LLaMA tokenizer in the format `[BLISS_{BlissID}]` and initializes its input and output embeddings.
   * Fine-tunes the model using `fine_tuning_sentences` and `validation_sentences`, optimizing both the new token's embeddings and the attention layers.
   * Saves the fine-tuned model to:
     `~/projects/ctb-whkchun/s2_bliss_LLMs/BLISS_{BlissID}`
   * Evaluates performance on `testing_positive_context_sentences`, `testing_negative_context_sentences`, and `testing_text_generation_prompts`.

3. **Utility script:** [`jobs/bliss-gloss/disambiguation/utils.py`](../jobs/bliss-gloss/disambiguation/utils.py)
   * Contains helper functions used within the main script.

4. **Job script:** [`jobs/bliss-gloss/multi-tokens-phrase/job_qlora_cover_multi_glosses.sh`](../jobs/bliss-gloss/multi-tokens-phrase/job_qlora_cover_multi_glosses.sh)
   * Submits the main script as a batch job to the Alliance server that offers the GPU access.

### Prepare Scripts

1. Ensure the dataset file is correctly named and contains all required variables.
2. Upload all relevant scripts to the Alliance server via FTP, placing them in the same working directory.
3. Edit the last line of [`job_qlora_cover_multi_glosses.sh`](../jobs/bliss-gloss/multi-tokens-phrase/job_qlora_cover_multi_glosses.sh) to match:
   - The correct path to the main script.
   - The target Bliss ID.
   - The associated glosses.

---

## Step 3: Run Job

1. Copy the job script to your `~/scratch` directory on the Alliance server.
2. Submit the job using SLURM:

    ```bash
    sbatch job_qlora_cover_multi_glosses.sh
    ```
