# python add_action_indicator_8993.py <data_dir> <action_indicator_id_pairs_json> <input_embedding_init_type> <output_embedding_init_type>
# python add_action_indicator_8993.py ~/bliss_gloss/integrate_grammar_indicators/data/ ~/bliss_gloss/integrate_grammar_indicators/data/action_indicator_id_pairs.json fixed

# This script adds the action indicator symbol token, with BCI-AV-ID 8993, to the model and fine-tunes the model
# with the new token. At the end, it evaluates the model to check if the new token is correctly learned.

# input_embedding_init_type defines how the input embedding is initialized. It can be one of the following:
# - "random": Use a random embedding as the input embedding for the new token.
# - "fixed": Use the input embedding of a fixed token, e.g., " to", as the input embedding for the new token.
# - "diff": Use the difference of the input embedding between the verb form and noun form as the input embedding
#   for the new token.

# output_embedding_init_type defines how the output embedding is initialized. It can be one of the following:
# - "random": Use a random embedding as the output embedding for the new token.
# - "fixed": Use the output embedding of a fixed token, e.g., " to", as the output embedding for the new token.
# - "diff": Use the difference of the output embedding between the verb form and noun form as the output embedding
#   for the new token.
# - "calc": Calculate the output embedding based on the context sentences that include the new token.

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

import os
import sys
import time
import json
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from utils import (
    calc_output_embedding,
    calc_embeddings_diff,
    evaluate_new_token
)
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
from dataset_to_as_infinitive_marker import positive_context_sentences, negative_context_sentences  # noqa: E402

if len(sys.argv) != 5:
    print("Usage: python add_action_indicator_8993.py <data_dir> <action_indicator_id_pairs_json> <input_embedding_init_type> <output_embedding_init_type>")
    sys.exit(1)

data_dir = sys.argv[1]
action_indicator_id_pairs_json = sys.argv[2]
input_embedding_init_type = sys.argv[3]  # "random" or "fixed" or "diff"
output_embedding_init_type = sys.argv[4]  # "random" or "fixed" or "diff" or "calc"

if input_embedding_init_type not in ["random", "fixed", "diff"]:
    print("Error: input_embedding_init_type must be one of 'random', 'fixed', or 'diff'")
    sys.exit(1)

if output_embedding_init_type not in ["random", "fixed", "diff", "calc"]:
    print("Error: output_embedding_init_type must be one of 'random', 'fixed', 'diff', or 'calc'")
    sys.exit(1)

# Set to True to initialize the new token's input and output embeddings with the same embeddings of a fixed token
FIXED_TOKEN = " to"

# Constants for building the dataset
TOKEN_TEMPLATE = "[BLISS_{bciAvId}]"

# Constants for adding a new symbol token
LEARNING_RATE = 0.0003
EPOCHS = 10
BATCH_SIZE = 20

bliss_id = 8993
dataset_file = os.path.join(data_dir, f"initial_{bliss_id}.py")

if not os.path.exists(dataset_file):
    print(f"Error: Initial dataset file '{dataset_file}' does not exist.")
    sys.exit(1)

print(f"Parameters - Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}\n")

# Track the total running time of this script
start_time = time.time()

# Read and execute the file to extract the target variables
namespace = {}
with open(dataset_file, "r") as f:
    initial_dataset = f.read()
exec(initial_dataset, namespace)

# Get datasets and process them
training_sentences = namespace["training_sentences"]
validation_sentences = namespace["validation_sentences"]
testing_predictions_sentences = namespace["testing_predictions_sentences"]
testing_text_generation_prompts = namespace["testing_text_generation_prompts"]

###############################
# Start of adding a new symbol token to the model

# Load the processed bliss IDs
root_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/"
base_model_dir = os.path.join(root_model_dir, "1_single_gloss_model")
adapter_dir = os.path.join(root_model_dir, "29_adapter_14443")
save_adapter_dir = os.path.expanduser("~") + f"/projects/ctb-whkchun/s2_bliss_LLMs/integrate_grammar_indicators/8993/input_{input_embedding_init_type}_output_{output_embedding_init_type}/"

new_token = TOKEN_TEMPLATE.format(bciAvId=bliss_id)
dtype = torch.bfloat16

# Load the tokenizer from the adapter directory
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded from the adapter model directory: {adapter_dir}")

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Note: Must resize the model vocaubulary size before loading the adapter because the vocabulary
# size of the base model is less than the size of the tokenizer loaded from the adapter directory
# as new tokens were added to the adapter.
model.resize_token_embeddings(len(tokenizer))

# Load latest adapter that includes the latest weights and embeddings for newly added tokens.
model = PeftModel.from_pretrained(model, adapter_dir)
print(f"Loaded latest adapter weights from {adapter_dir}")
# End of loading the model and tokenizer

# Add the new token to the tokenizer
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"New token '{new_token}' added to the model with ID: {new_token_id}. Embeddings will be initialized later.")

# Prepare embeddings for the initialization of the new token
if input_embedding_init_type == "fixed" or output_embedding_init_type == "fixed":
    initial_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(FIXED_TOKEN))[0]
    print(f"Token ID of the fixed token '{FIXED_TOKEN}': {initial_token_id}")

    input_embedding_of_fixed_token = model.get_input_embeddings().weight.data[initial_token_id]
    output_embedding_of_fixed_token = model.get_output_embeddings().weight.data[initial_token_id]

if input_embedding_init_type == "diff" or output_embedding_init_type == "diff":
    model_dir_for_dataset = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
    pairs_data = json.load(open(action_indicator_id_pairs_json, "r"))

    # Load model and tokenizer for building the dataset
    tokenizer_for_calc = AutoTokenizer.from_pretrained(model_dir_for_dataset)
    model_for_calc = AutoModelForCausalLM.from_pretrained(model_dir_for_dataset)

    # Set the model to evaluation mode
    model.eval()

    input_embedding_of_diff, output_embedding_of_diff = calc_embeddings_diff(
        model_for_calc, tokenizer_for_calc, pairs_data
    )

    # Release the model and tokenizer to free up memory
    del model_for_calc
    del tokenizer_for_calc

    # Clean up GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Run garbage collection
    gc.collect()

    print(f"DIFF: Use input and output embeddings of the difference between verb and noun forms for the new token '{new_token}'")

# Init the input embedding of the new token
if input_embedding_init_type == "random":
    new_token_input_embedding_before = torch.randn(model.get_input_embeddings().weight.data[0].shape, dtype=dtype)
    print(f"RANDOM: Use a random input embedding for the new token '{new_token}'")
elif input_embedding_init_type == "fixed":
    new_token_input_embedding_before = input_embedding_of_fixed_token.clone()
    print(f"FIXED: Use input embedding of the fixed token '{FIXED_TOKEN}' as the input embedding for the new token '{new_token}'")
elif input_embedding_init_type == "diff":
    new_token_input_embedding_before = input_embedding_of_diff.clone()
    print(f"DIFF: Use input embedding of the difference between verb and noun forms as the input embedding for the new token '{new_token}'")

# Init the input embedding of the new token
if output_embedding_init_type == "random":
    new_token_output_embedding_before = torch.randn(model.get_output_embeddings().weight.data[0].shape, dtype=dtype)
    print(f"RANDOM: Use a random output embedding for the new token '{new_token}'")
elif output_embedding_init_type == "fixed":
    new_token_output_embedding_before = output_embedding_of_fixed_token.clone()
    print(f"FIXED: Use output embedding of the fixed token '{FIXED_TOKEN}' as the output embedding for the new token '{new_token}'")
elif output_embedding_init_type == "diff":
    new_token_output_embedding_before = output_embedding_of_diff.clone()
    print(f"DIFF: Use output embedding of the difference between verb and noun forms as the output embedding for the new token '{new_token}'")
elif output_embedding_init_type == "calc":
    initial_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(FIXED_TOKEN))[0]
    new_token_input_embedding_before = model.get_input_embeddings().weight.data[initial_token_id]
    print(f"CALC_OUTPUT_EMBEDDING: Use the input embedding of the fixed token '{FIXED_TOKEN}'({initial_token_id}) as the input embedding for the new token '{new_token}'")
    # Calculate the output embedding based on the context sentences
    new_token_output_embedding_before = calc_output_embedding(
        model, tokenizer, positive_context_sentences, negative_context_sentences, initial_token_id, dtype=dtype
    )
    print(f"CALC_OUTPUT_EMBEDDING: Use calculated output embedding for the new token '{new_token}'")

with torch.no_grad():
    model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()
    model.get_output_embeddings().weight.data[new_token_id] = new_token_output_embedding_before.clone()

# Preprocess the quantized model for QLoRA Training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
# Convert to PeftModel if not already
if not isinstance(model, PeftModel):
    model = get_peft_model(model, peft_config)


def tokenize_func(dataset):
    # Prepare Dataset
    tokenized = tokenizer(
        dataset["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_dataset = (
    Dataset.from_dict({"text": training_sentences})
    .map(tokenize_func, batched=True)
)

validation_dataset = (
    Dataset.from_dict({"text": validation_sentences})
    .map(tokenize_func, batched=True)
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=save_adapter_dir,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=3,
    gradient_checkpointing=True,  # Reduces memory usage
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,    # Use mixed precision training
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=1,
    eval_strategy="steps",
    eval_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,   # False for loss (lower is better), True for metrics like accuracy
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

end_time_prepare = time.time()
elapsed_time = end_time_prepare - start_time
print(f"Preparation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Train
print("\nStarting training...")
trainer.train()
print("Training completed!")

end_time_fine_tuning = time.time()
elapsed_time = end_time_fine_tuning - end_time_prepare
print(f"Fine-tuning time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Save Model
model.save_pretrained(save_adapter_dir)
tokenizer.save_pretrained(save_adapter_dir)
print(f"Saved new adapter to {save_adapter_dir}")

# Compare embeddings of the new token before and after fine-tuning
new_token_input_embedding_after = model.get_input_embeddings().weight.data[new_token_id].clone()
input_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_input_embedding_before.to(model.device), new_token_input_embedding_after.to(model.device), dim=0)
print(f"Cosine Similarity of the new token input embedding before and after: {input_embedding_similarity:.4f}")
distance = torch.norm(new_token_input_embedding_before.to(model.device) - new_token_input_embedding_after.to(model.device), p=2)
print(f"Euclidean Distance of the new token input embedding before and after: {distance.item():.4f}")

new_token_output_embedding_after = model.get_output_embeddings().weight.data[new_token_id].clone()
output_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_output_embedding_before.to(model.device), new_token_output_embedding_after.to(model.device), dim=0)
print(f"Cosine Similarity of the new token output embedding before and after: {output_embedding_similarity:.4f}")
distance = torch.norm(new_token_output_embedding_before.to(model.device) - new_token_output_embedding_after.to(model.device), p=2)
print(f"Euclidean Distance of the new token output embedding before and after: {distance.item():.4f}\n")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")

# Set the model to evaluation mode for inference
model.eval()

# Evaluate the results
expected_predictions = [" to", "["]
target_tokens = [tokenizer.tokenize(token)[0] for token in expected_predictions]
target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
print(f"Target tokens: {target_tokens}; Target token IDs: {target_token_ids}")

evaluation_results = evaluate_new_token(
    model, tokenizer, new_token, testing_predictions_sentences, testing_text_generation_prompts,
    target_tokens, target_token_ids, new_token_id
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
