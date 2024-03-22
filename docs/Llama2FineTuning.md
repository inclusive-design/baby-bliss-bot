# Experiment with Llama2 model

The experiment leveraged the [7B parameter Llama2 model pretrained by Meta](https://huggingface.co/meta-llama/Llama-2-7b-hf),
converted for the seamless use of the Hugging Face Transformers format. This model is choosen as a starting
point because it requires less training time and GPU resources compared to its larger counterparts, while it
potentially sacrifies some capability. Additionally, the Hugging Face Transformers format is selected because
of its extensive community support and standardized APIs.

## Download Llama-2-7b-hf to Cedar

1. Request access to Llama2 models on [the Meta website](https://llama.meta.com/llama-downloads/);
2. Followed the instructions on [the Hugging Face website](https://huggingface.co/meta-llama/Llama-2-7b-hf)
to request the access to its Llama2 model;
3. Request a hugging face access token on [this page](https://huggingface.co/settings/tokens);
4. Login to the Cedar cluster;
5. Create a "llama" directory and run these commands to download the model:

```
mkdir llama2
cd llama2

// Load git-lfs first for downloading via Git large file storage
module load git-lfs/3.3.0
git lfs install

git clone https://{hugging_face_id}:{hugging_face_access_token}@huggingface.co/meta-llama/Llama-2-7b-hf

// Fetch git large files in the repo directory
cd Llama-2-7b-hf
git lfs fetch
```

6. Copy the content of [`requirements.txt`](https://github.com/facebookresearch/llama/blob/main/requirements.txt)
for setting up the Llama2 models into a new file named `requirements-llama2.txt` in the "llama" directory.

## Use the Original Llama2 Model

In the [`jobs/original_use`](../jobs/Llama2/original_use) directory, there are two scripts:

* original_use_7b_hf.py: The script that loads the downloaded model and tokenizer to perform text generation,
word predictions and making inferences
* job_original_use_7b_hf.sh: The job script submitted to Cedar to run `original_use_7b_hf.py`

Note that the job script must be copied to the user's `scratch` directory and is submitted from there using
the `sbatch` command.

FTP scripts above to the cedar cluster in the users `llama2/original_use` directory. Run the following command to
submit the job.

```
cp llama2/original_use/job_original_use_7b_hf.sh scratch/.
cd scratch
sbatch job_original_use_7b_hf.sh
```

The result is written to the `llama2/original_use/result.txt`.

## Fine-tune the Llama2 Model

In the [`jobs/finetune`](../jobs/Llama2/finetune) directory, there are these scripts:

* bliss.json: The dataset that converts English text to the structure in the Conceptual Bliss
* finetune_7b_hf.py: The script that fine-tunes the downloaded model
* job_finetune_7b_hf.sh: The job script submitted to Cedar to run `finetune_7b_hf.py`

FTP scripts above to the cedar cluster in the users `llama2/finetune` directory. Run the following command to
submit the job.

```
cp llama2/finetune/job_finetune_7b_hf.sh scratch/.
cd scratch
sbatch job_finetune_7b_hf.sh
```

The fine-tuning script does:

1. Create an instruction dataset using `bliss.json`. This dataset contains bi-directional conversion between
English and Conceptual Bliss. 
2. Use the dataset to fine-tune the Llama2 model. See `finetune_7b_hf.py` about the fine-tuning parameters.
3. Evaluate the fine-tuned model by giving instructions fine-tuned for, along with a few sentences for language
conversion.

Please note that due to the relatively small size of the dataset derived from bliss.json, the fine-tuning script
was run four times, adjusting the epoch number in the script from 1 to 4. As a result, 4 models were generated
corresponding to the different epoch counts.

## Evaluate the Fine-tuned Model

This section describes how to evaluate a fine-tuned model with instructions and input sentences.

In the [`jobs/finetune`](../jobs/Llama2/finetune) directory, there are these scripts:

* eval_7b_hf.py: The script that fine-tunes the downloaded model. Common variables to adjust:
  * `model_dir`: The location of the model directory
  * `instruction`: At the bottom of the script, define the instruction part in a prompt
  * `input`: At the bottom of the script, define the sentence to be converted
* job_eval_7b_hf.sh: The job script submitted to Cedar to run `eval_7b_hf.py`

FTP scripts above to the cedar cluster in the users `llama2/finetune` directory. Run the following command to
submit the job.

```
cp llama2/finetune/job_eval_7b_hf.sh scratch/.
cd scratch
sbatch job_eval_7b_hf.sh
```

## Evaluate the Generated Sentences from the Fine-tuned Model

This section describes how to evaluat the generated sentences and compare them with original or expected sentences.
It evaluates the generated sentence in these aspects:

* Semantic Coherence
* Novelty and Creativity
* Fluency and Readability

In the [`jobs/finetune`](../jobs/Llama2/finetune) directory, there are these scripts:

* eval_generated_sentence.py: The script that fine-tunes the downloaded model. Common variables to adjust:
  * `sentence_orig`: The original sentence
  * `sentence_expected`: The expected sentence
  * `sentence_generated`: The sentence generated by the fine-tuned model
* job_eval_generated_sentence.sh: The job script submitted to Cedar to run `eval_generated_sentence.py`

FTP scripts above to the cedar cluster in the users `llama2/finetune` directory. Run the following command to
submit the job.

```
cp llama2/finetune/job_eval_generated_sentence.sh scratch/.
cd scratch
sbatch job_eval_generated_sentence.sh
```

## Conclusion

Although the fine-tuning uses a fairly small dataset, the fine-tuned model performs pretty well in converting English
and Conceptual Bliss sentence structure, especially with the two-epochs and three-epochs models.

## References

[Llama2 in the Facebook Research Github repository](https://github.com/facebookresearch/llama)
[Llama2 fine-tune, inference examples](https://github.com/facebookresearch/llama-recipes)
[Llama2 on Hugging Face](https://huggingface.co/docs/transformers/model_doc/llama2)
[Use Hugging Face Models on Cedar Clusters](https://docs.alliancecan.ca/wiki/Huggingface)
