# Steps to Integrate Bliss Symbols into LLaMA

This document outlines the process for integrating Bliss Meaning Symbols into a LLaMA language model by introducing a new token for each Bliss symbol. These symbols represent specific concepts such as “tree,” “house,” and others. Each token is formatted as [BLISS_{blissID}].

The working directory for all steps is: [`job/bliss-gloss/integrate-bliss-symbols`](../jobs/bliss-gloss/integrate-bliss-symbols).

**Note:** Some scripts use a Bliss symbol explanation file [`bliss_symbol_explanations.json`](../jobs/bliss-gloss/data/bliss_symbol_explanations.json) to look up symbol glosses and explanations. This file is a copy from the [`inclusive-design/adaptive-palette`](https://github.com/inclusive-design/adaptive-palette/blob/main/public/data/bliss_symbol_explanations.json) repository.

## Step 1: Add Single-Token Gloss Symbols

This step has two sub-steps:

1. Gloss Clean-up
2. Add Single-Token-Gloss Symbols

**Working directory**: [`job/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols)

### Step 1.1. Gloss Clean-up

Symbol glosses are provided in a spreadsheet. They have special markers and separators. This step is to normalize them into individual glosses that make sense for processing.

The script performs the following processing steps:

1. Handle Special Glosses: Checks if an item's symbol id matches a predefined list of special cases (e.g., punctuation, numbers, single letters). If a match is found, uses the predefined gloss value directly, skipping all other processing for that item.

2. Initial Cleanup: For regular items, replace all underscores (_) with spaces to create more readable text.

3. Remove Suffix "(OLD)": If a description ends with "(OLD)", this suffix and any preceding whitespace are removed.

4. Handle Plurals (s): Split the description into individual glosses by commas. If any gloss contains "(s)" (e.g., "glove(s)"), it expands it into two separate glosses: a singular form ("glove") and a plural form ("gloves").

    Process Verb Marker -(to): If the original description ends with -(to), the script flags it, removes the suffix, and then prepends "to " to every resulting gloss for that entry (e.g., "advance,go-(to)" becomes "to advance", "to go").

### Step 1.2 Add Single-Token-Gloss Symbols

This step identifies and adds symbols that meet specific criteria.

#### Criteria

A Bliss symbol is added if:

1. It has **only one gloss**.
2. That gloss is **a single token** according to the LLaMA tokenizer.

#### Description

The script performs the following steps:

1. Iterates through the Bliss dictionary.
2. Checks if each symbol meets the above criteria.
3. For qualifying symbols:
  - Adds a new token in the format of `[BLISS_{blissID}_start]`.
  - Copies the input and output embeddings from the original gloss token to the new token.

**Total tokens added in this step:** 2369

#### Script Details

* Script: [`add_single_token_gloss_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/add_single_token_gloss_symbols.py)

* Usage:

```bash
python add_single_token_gloss_symbols.py {input_gloss_json} {output_file_with_added_id} {output_file_with_not_added_id}
```

* Example:

```bash
python ~/bliss_gloss/add_single_token_gloss_symbols.py ./data/bliss_gloss_cleaned.json ./outputs/bliss_ids_added.json ./outputs/bliss_ids_not_added.json
```

* Input:

**Bliss Dictionary JSON**: [`bliss_gloss_cleaned.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/data/bliss_gloss_cleaned.json)

* Output:

* **The output model** is saved in `~/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/single_gloss_model`.
* **Added symbols list**: [`output/bliss_ids_added.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/output/bliss_ids_added.json)
* **Skipped Symbols List** (symbols not added): [`/output/bliss_ids_not_added.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/output/bliss_ids_not_added.json)

## Step 2: Find most frequently used symbols

**Working directoy** [`jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/)

Two methods were used to identify frequently used symbols:

1. The Bliss standard board

  * Symbols in the board are extracted to [`standard_board_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/standard_board_symbols.json).
  * A total of **473 symbols** are extracted: 
    * Note: Although "was", "were", and "had" appear on the standard board, they are not included in the extracted list because they do not have unique symbols. Instead, they are represented by variations of existing concepts: **"being (state of)"** (ID: 24443) for "was/were", and **"holding"** (ID: 24912) for "had".

2. Frequency-Tagged Spreadsheet (by Mats Lundälv)

Mats Lundälv, a former chairman of BCI, sorted the Bliss words into groups for frequency analysis. Group 1 is higher frequency, Group 5 is lower. Ungrouped are even lower. 

  * [The spreadsheet](https://docs.google.com/spreadsheets/d/1E4H45LYffyWKT0Devp32wfB1NOPBnPEp/edit?usp=drive_link&ouid=111581944030695000923&rtpof=true&sd=true).
  * Symbols are grouped by its tag number and reported in [this file](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/frequency_tagged_symbols.json).
  * Group counts:
    * Group 1: **644 symbols**
    * Group 2: **127 symbols**
    * Group 3: **141 symbols**
    * Group 4: **95 symbols**
    * Group 5: **13 symbols**
    * Ungrouped: **4815 symbols**

### Overlap Analysis

Based on the above sources:
* **417 symbols** appear in both the standard board and frequency group 1
* **56 symbols** are only in the standard board
* **227 symbols** are only in frequency group 1

So, the primary focus is the **288 overlapping symbols**.

### Script Details

#### Extract symbols from Bliss standard board

* Script: [`get_standard_board_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/get_standard_board_symbols.py)

* Usage: 

```bash
python get_standard_board_symbols.py <bliss_standard_chart_json> <bliss_symbol_explanations_json> <output_symbols_file>")
```

* Example:

```bash
python get_standard_board_symbols.py ../../../../../adaptive-palette/public/palettes/bliss_standard_chart.json ../../data/bliss_symbol_explanations.json ./output/standard_board_symbols.json
```

* Output: [`output/standard_board_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/standard_board_symbols.json)

* Note: <bliss_standard_chart_json> must point to the json file that contains other JSON files that the "branchTo" field refers to.

#### Extract symbols by their frequency group

* Script: [`get_frequency_tagged_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/get_frequency_tagged_symbols.py)

* Usage:

```bash
python get_frequency_tagged_symbols.py <frequency_tagged_symbols_csv> <output_tagged_symbols_json>
```

* Example:

```bash
python get_frequency_tagged_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
```
* Output: [`output/frequency_tagged_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/frequency_tagged_symbols.json)

#### Compare symbols in the standard board with symbols in the frequency group 1

* Script: [`compare_most_frequent_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/compare_most_frequent_symbols.py)

* Usage:

```bash
python compare_most_frequent_symbols.py <bliss_symbol_explanations_json> <output_overlapped_symbols_json> <output_only_in_standard_board_json> <output_only_in_tag1_json>
```

* Example:

```bash
python compare_most_frequent_symbols.py ../../data/bliss_symbol_explanations.json ./output/overlapped_symbols.json ./output/only_in_standard_board.json ./output/only_in_tag1.json
```

* Output file:
  * [`output/overlapped_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/overlapped_symbols.json): In both standard board and frequency group 1
  * [`output/only_in_standard_board.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/only_in_standard_board.json): Only in standard board
  * [`output/only_in_tag1.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/only_in_tag1.json): Only in frequency group 1

## Step 3: Identify Missing High-Frequency Symbols

**Working directoy**: [`jobs/bliss-gloss/integrate-bliss-symbols/3-find-frequently-used-symbols/`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-frequently-used-symbols/).

This step compares **417 most frequently used symbols** from Step 2 aginst the list of symbols already been added in the step 1 and finds high-priority Bliss symbols that were not added during Step 1.

**230 symbols** were found to be missing. 

They are written to [`output/missing_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-frequently-used-symbols/output/missing_symbols.json).

### Script Details

* Script: [`find_missing_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-frequently-used-symbols/find_missing_symbols.py)

* Usage:

```bash
python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json>
```

* Example:
```bash
python find_missing_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
```

* Output file:
  * [`output/missing_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-frequently-used-symbols/output/missing_symbols.json): 160 missing symbols

## Step 4: Add Symbols Not Included in Step 1

**Working directory**: [`jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/)

This step focuses on adding Bliss symbols that were not included in **Step 1**.

To optimize the process, we begin with the 160 most frequently used missing symbols and continue with the remaining symbols.

The procedure outlined below should be repeated for each new symbol being added to the model.

### Step 4.1 Build Dataset for a Symbol

This stage prepares training and evaluation data for a specific Bliss symbol. We'll use the symbol with BCI-AV-ID `12321` (glosses: `"a"`, `"an"` `"any"`) as an example.

1. **Generate the initial dataset** 

Use a large language model (e.g., ChatGPT, Claude, DeepSeek) to create a initial dataset containing diverse contexts for the glosses. See [example prompts](../../multi-tokens-phrase/data/prmpts).
  
Your dataset should define 4 Python lists:
  * `positive_context_sentences`: Complete sentences with strong contextual cues that naturally lead to the target gloss.
  * `negative_context_sentences`: Partial sentences that should **not** lead to the target gloss. These, together with `positive_context_sentences`, are used to compute the output embedding of the new token.
  * `fine_tuning_sentences`: Complete sentences that explicitly include at least one target gloss. These will be used to fine-tune the token embedding.
  * `testing_text_generation_prompts`: Partial sentences containing the placeholder string `{placeholder}`, where the gloss or token is expected to appear. Used to evaluate model performance after fine-tuning.

**Note**: The data file must be saved in the [`data/`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/data/) directory, using the naming convention `initial_{bci_av_id}.py`. The script will automatically look for a file at this location with a matching filename.

2. **Process the dataset** 

Run the `build_dataset.py` script. This script transforms the initial dataset into a format ready for training and evaluation:

* **Positive context sentences** -> `training_positive_context_sentences` & `testing_positive_context_sentences`:
  1. Replace curly quotes (`’`) with straight quotes (`'`)
  2. Truncate each sentence at the last occurrence of any target gloss
  3. Discard truncated sentences if shorter than `MIN_PARTIAL_SENTENCE_LENGTH`
  4. Check the prediction rank of the first token of each gloss with the truncated sentence. Discard the truncated sentences if none of the gloss prediction ranks are lower than `RANK_POSITIVE_THRESHOLD.`
  5. Split 80% for training and 20% for testing

* **Negative context sentences** -> `training_negative_context_sentences` & `testing_negative_context_sentences`:
  1. Replace curly quotes (`’`) with straight quotes (`'`)
  2. Check the prediction rank of the first token of each gloss in the partial sentences. Discard the sentences if any one of the prediction ranks is lower than `RANK_NEGATIVE_THRESHOLD.`
  3. Split 80% for training and 20% for testing

* **Fine-tuning sentences** -> `fine_tuning_sentences` & `validation_sentences`:
  1. Replace curly quotes (`’`) with straight quotes (`'`)
  2. Replace the glosses of the current `bci_av_id` with the corresponding token
  3. Replace glosses of *existing* BCI-AV-IDs with their corresponding tokens to ensure proper inter-symbol learning
  4. Shuffle the dataset
  5. Split 80% for training and 20% for validation

* **Testing prompts** -> `testing_text_generation_prompts`:
  1. Replace curly quotes (`’`) with straight quotes (`'`)
  2. Remove any space before the `{placeholder}`

**Constants and Their Default Values**:

| Constant                        | Value               |
| ------------------------------- | ------------------- |
| `MIN_PARTIAL_SENTENCE_LENGTH`   | 40                  |
| `RANK_POSITIVE_THRESHOLD`       | 100                 |
| `RANK_NEGATIVE_THRESHOLD`       | 1000                |
| `TOKEN_TEMPLATE`                | `[BLISS_{bciAvId}]` |
| `MIN_NUM_OF_TRAINING_SENTENCES` | 100                 |

### Step 4.2 Add the New Symbol Token

This step incrementally integrates symbol tokens into the model and fine-tunes their representations. Each symbol is added one at a time, with the integration of each new symbol building upon the representations of previously incorporated symbols. This process enables the model to learn the contextual meaning and usage of each symbol.

1. **Compute the initial output embedding** of the new symbol token using training context sentences: `training_positive_context_sentences` and `training_negative_context_sentences`.
2. **Compute the initial input embedding** as the average of all token embeddings across the target glosses.
3. **Add the new symbol token to the LLaMA tokenizer** in the format `[BLISS_{bci_av_id}]` and initializes its input and output embeddings.
4. **Fine-tune the model** using the training and validation data: `fine_tuning_sentences` and `validation_sentences`. This optimizes both the new token's embeddings and the attention layers.
5. Saves the adapter to `~/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/{sequence_id}_adapter_{bci_av_id}`
6. **Evaluate performance** of the new token on testing context sentences and text generation prompts: `testing_positive_context_sentences`, `testing_negative_context_sentences`, and `testing_text_generation_prompts`.

### Script Details

* **Integration Script**: [`add_new_symbol_token.py`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/add_new_symbol_token.py)

* **Usage**:
```bash
python add_new_symbol_token.py <bliss_id> <glosses> <data_dir> <processed_bliss_ids_json>
```

* **Example**:
```bash
python add_new_symbol_token.py 12327 '["afraid", "frightened", "scared"]' ~/bliss_gloss/4-add-missing-symbols/data/ ~/bliss_gloss/4-add-missing-symbols/data/bliss_ids_added.json
```

* **Output**:
  * **Adapter** 
    
    Saved on the Alliance server at `~/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/{sequence_id}_adapter_{bci_av_id}`

  * **Processed dataset**

    [`/data/dataset_{bci_av_id}_{gloss_1}_{gloss_2}_..._{gloss_N}.py`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/data). For example, for BCI-AV-ID 12321 with glosses "a", "an", and "any", the processed dataset file is: [`./data/dataset_12321_a_an_any.py`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/data/dataset_12321_a_an_any.py)

  * **The fine-tuning log file**

    [`/logs/{sequence_id}_{bci_av_id}`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/logs)

  * **Updated Bliss Registry**

    [`bliss_ids_added.json`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/data/bliss_ids_added.json`)

Note: The job script `job_add_new_symbol_token.sh` is responsible for submitting the integration script above to the Alliance server for execution.

## Step 5: Periodic Model Evaluation

Step 4 generates multiple adapters. Periodically evaluate previously added tokens using newer adapters to ensure they continue to perform as expected.

### Script Details

* **Evaluation Script**: [`test_model.py`](../jobs/bliss-gloss/integrate-bliss-symbols/4-add-missing-symbols/test_model.py)

* **Description**:  
  This script loads the base model along with a specified adapter, retrieves the test data from the dataset, and evaluates the test data using the loaded model.

* **Usage**:
  ```bash
  python test_model.py <bliss_id> <glosses> <data_dir> <adapter_dir_name>
  ```

* **Example**:
```bash
python test_model.py 12356 '["air", "atmosphere"]' 'data' '19_adapter_12869'
```

* **Output**:
  The script prints out the evaluation results to the console.
