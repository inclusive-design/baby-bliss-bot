# python add_new_symbol_token.py <bliss_id> <glosses> <data_dir> <processed_bliss_ids_json>
# python add_new_symbol_token.py 12327 '["afraid", "frightened", "scared"]' ~/bliss_gloss/4-add-missing-symbols/data/ ~/bliss_gloss/4-add-missing-symbols/data/bliss_ids_added.json

# This script adds a new token to the model for a given Bliss ID. It caluculates the output
# embedding of the new token, initialize its input embedidng to the mean of all composing tokens
# in its glosses. Then it fine-tunes the model using QLoRA and evaluates the model's performance
# of this new token.

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

# The path to the best model is saved in the Trainer state, and can be accessed using: trainer.state.best_model_checkpoint
# This path is also saved in "output_dir/checkpoint-<step>/trainer_state.json" -> "best_model_checkpoint" path

import os
import sys
import time
import json
import re
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from utils import (
    get_output_file_location,
    evaluate_new_token
)

# Constants for building the dataset
MIN_PARTIAL_SENTENCE_LENGTH = 40
RANK_POSITIVE_THRESHOLD = 100
RANK_NEGATIVE_THRESHOLD = 1000
TOKEN_TEMPLATE = "[BLISS_{bciAvId}]"
MIN_NUM_OF_TRAINING_SENTENCES = 100

# Constants for adding a new symbol token
LEARNING_RATE = 0.0003
EPOCHS = 3
BATCH_SIZE = 10

if len(sys.argv) != 5:
    print("Usage: python add_new_symbol_token.py <bliss_id> <glosses> <data_dir> <processed_bliss_ids_json>")
    sys.exit(1)

bliss_id = sys.argv[1]
glosses = json.loads(sys.argv[2])
data_dir = sys.argv[3]
processed_bliss_ids_json = sys.argv[4]
initial_dataset_file = get_output_file_location(data_dir, bliss_id, glosses)

if not os.path.exists(initial_dataset_file):
    print(f"Error: Initial dataset file '{initial_dataset_file}' does not exist.")
    sys.exit(1)

# Track the total running time of this script
start_time = time.time()

# START to build the dataset
model_dir_for_dataset = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"

# Load model and tokenizer for building the dataset
tokenizer = AutoTokenizer.from_pretrained(model_dir_for_dataset)

# Initialize the new token embeddings
# Add a space before each gloss to ensure they will be tokenized with a start of the word token
glosses_in_words = [" " + gloss for gloss in glosses]
print(f"Glosses in words: {glosses_in_words}")
target_tokens = [tokenizer.tokenize(token)[0] for token in glosses_in_words]
target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
print(f"Target tokens: {target_tokens}; Target token IDs: {target_token_ids}")

# Append the new token ID into the existing processed ID list. The glosses in the list will be replaced
# by their corresponding tokens when processing the fine-tuning sentences.
existing_id_gloss_list = json.load(open(processed_bliss_ids_json, "r"))
existing_id_gloss_list[bliss_id] = glosses

# Read and execute the file to extract the target variables
namespace = {}
with open(initial_dataset_file, "r") as f:
    initial_dataset = f.read()
exec(initial_dataset, namespace)

# Get datasets and process them
training_positive_context_sentences = namespace["training_positive_context_sentences"]
training_negative_context_sentences = namespace["training_negative_context_sentences"]
testing_positive_context_sentences = namespace["testing_positive_context_sentences"]
testing_negative_context_sentences = namespace["testing_negative_context_sentences"]
fine_tuning_sentences = namespace["fine_tuning_sentences"]
validation_sentences = namespace["validation_sentences"]
testing_text_generation_prompts = namespace["testing_text_generation_prompts"]

# Release the tokenizer to free up memory
del tokenizer

# Clean up GPU memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Run garbage collection
gc.collect()

# END of building the dataset
###############################

###############################
# Start of adding a new symbol token to the model

# Load the processed bliss IDs
root_script_dir = os.path.expanduser("~") + "/bliss_gloss/4-add-missing-symbols/"
root_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/integrate_bliss_symbols/"
base_model_dir = os.path.join(root_model_dir, "1_single_gloss_model")
new_token = TOKEN_TEMPLATE.format(bciAvId=bliss_id)
dtype = torch.bfloat16

# Determine next version number
existing_dirs = [d for d in os.listdir(root_model_dir) if os.path.isdir(os.path.join(root_model_dir, d))]
model_versions = [int(re.match(r'^(\d+)_model_', d).group(1)) for d in existing_dirs if re.match(r'^\d+_model_', d)]
# The next version starts from 2 because 1 is used for the base model
next_version = max(model_versions) + 1 if model_versions else 2

# Create new model directory name
save_dir = f"{next_version}_model_{bliss_id}"
save_path = os.path.join(root_model_dir, save_dir)

# Load the model and tokenizer
if model_versions:
    latest_model = [dir for dir in existing_dirs if dir.startswith(f"{max(model_versions)}_model_")][0]
    latest_model_dir = os.path.join(root_model_dir, latest_model)
else:
    # First time running, load from base model
    latest_model_dir = base_model_dir

tokenizer = AutoTokenizer.from_pretrained(latest_model_dir)
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit Quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=dtype,
#     bnb_4bit_use_double_quant=True,
# )

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    # quantization_config=bnb_config,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
)
print(f"Loaded model and tokenizer from the latest model dir: {latest_model_dir}")

# Load latest adapter that includes the latest weights and embeddings for newly added tokens.
# if model_versions:
#     model = PeftModel.from_pretrained(model, latest_model_dir)
#     print(f"Loaded latest adapter weights from {latest_model_dir}")
# End of loading the model and tokenizer

# Add the new token to the tokenizer
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"New token '{new_token}' added to the model with ID: {new_token_id}. Embeddings will be initialized later.")

# Calculate the output embedding for the new token
# new_token_output_embedding_before = calc_output_embedding(model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids, dtype)
# print("Calculated output embedding for the new token")
new_token_output_embedding_before = torch.randn(model.get_output_embeddings().weight.data[0].shape, dtype=model.get_output_embeddings().weight.data.dtype)
print("Generated random output embedding for the new token")
model.get_output_embeddings().weight.data[new_token_id].copy_(new_token_output_embedding_before)
print(f"Output embedding of the new token '{new_token}' set to the calculated output embedding")

# new_token_input_embedding_before = get_average_input_embeddings(model, tokenizer, glosses_in_words)
new_token_input_embedding_before = torch.randn(model.get_input_embeddings().weight.data[0].shape, dtype=model.get_input_embeddings().weight.data.dtype)
print("Generated random input embedding for the new token")
# Reset the input embedding of the new token to the initial token
model.get_input_embeddings().weight.data[new_token_id].copy_(new_token_input_embedding_before)
print(f"Input embedding of the new token '{new_token}' set to the average input embedding of all constituent tokens of '{glosses_in_words}'")

# with torch.no_grad():
#     model.get_output_embeddings().weight[new_token_id] = new_token_output_embedding_before.clone()
#     print(f"Output embedding of the new token '{new_token}' set to the calculated output embedding")

#     new_token_input_embedding_before = get_average_input_embeddings(model, tokenizer, glosses_in_words)
#     # Reset the input embedding of the new token to the initial token
#     model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()
#     print(f"Input embedding of the new token '{new_token}' set to the average input embedding of all constituent tokens of '{glosses_in_words}'")

# # Preprocess the quantized model for QLoRA Training
# model = prepare_model_for_kbit_training(model)

model.enable_input_require_grads()
print("Enabled input embeddings to require gradients")

print(f"Model configuration: {model.config}")

input_embeddings_layer = model.get_input_embeddings()
# Freeze all input embeddings first
input_embeddings_layer.weight.requires_grad = False
# Then enable gradients for the new token's embedding
input_embeddings_layer.weight.data[new_token_id].requires_grad = True

output_embeddings_layer = model.get_output_embeddings()
# Freeze all output embeddings first
output_embeddings_layer.weight.requires_grad = False
# Then enable gradients for the new token's embedding
output_embeddings_layer.weight.data[new_token_id].requires_grad = True

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
    # Attention: q_proj, k_proj, v_proj, o_proj
    # FFN: gate_proj, up_proj, down_proj
    # Input embedding: embed_tokens
    # Output embedding: lm_head
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
)
# Convert to PeftModel if not already
if not isinstance(model, PeftModel):
    model = get_peft_model(model, peft_config)

# Explicitly set to training mode
model.train()
print("Model set to training mode")


def tokenize_func(dataset):
    # Prepare Dataset
    tokenized = tokenizer(
        dataset["text"],
        padding=True,
        max_length=128,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_dataset = (
    Dataset.from_dict({"text": fine_tuning_sentences})
    .map(tokenize_func, batched=True)
)

validation_dataset = (
    Dataset.from_dict({"text": validation_sentences})
    .map(tokenize_func, batched=True)
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=3,
    gradient_checkpointing=True,  # Reduces memory usage
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,    # Use mixed precision training
    max_grad_norm=1.0,   # Add gradient clipping
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="steps",
    save_steps=5,
    eval_strategy="steps",
    eval_steps=5,
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

print("Printing trainable parameters...")
model.print_trainable_parameters()
print("Printed trainable parameters.")

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

# Save merged model and tokenizer
model = model.merge_and_unload()
print("Merged LoRA weights into the base model")

# Save Model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved merged model and tokenizer to {save_path}")

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
print("Model set to evaluation mode")

# Evaluate the results
evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, glosses_in_words
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Append the newly processed bliss ID to the JSON file
with open(processed_bliss_ids_json, "w") as f:
    json.dump(existing_id_gloss_list, f, indent=4, ensure_ascii=False)
