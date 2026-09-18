# This script adds a new token to the model for a given Bliss ID. It caluculates the output
# embedding of the new token, initialize its input embedidng to the mean of all composing tokens
# in its glosses. Then it fine-tunes the model using QLoRA and evaluates the model's performance
# of this new token.

# python qlora_cover_multi_glosses.py <bliss_id> <glosses>
# python qlora_cover_multi_glosses.py 24918 '[" lowness", " shortness"]'

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

# The path to the best model is saved in the Trainer state, and can be accessed using: trainer.state.best_model_checkpoint
# This path is also saved in "output_dir/checkpoint-<step>/trainer_state.json" -> "best_model_checkpoint" path

import os
import sys
import importlib
import time
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from utils import create_training_data, calc_embeddings, evaluate_new_token, get_average_input_embeddings  # noqa: E402
# from data import user_dataset as user_dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import create_training_data, calc_embeddings, evaluate_new_token, get_average_input_embeddings  # noqa: E402


def calc_output_embedding(model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids, dtype):
    # Track the total running time of this script
    start_time = time.time()

    hidden_states, target_logits = create_training_data(
        model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
    )
    new_output_embedding = calc_embeddings(hidden_states, target_logits, dtype)

    end_time_calc_output_embedding = time.time()
    elapsed_time = end_time_calc_output_embedding - start_time
    print(f"Execution time for calculating output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")
    return new_output_embedding


if len(sys.argv) != 3:
    print("Usage: python qlora_cover_multi_glosses.py <bliss_id> <glosses>")
    sys.exit(1)

bliss_id = sys.argv[1]
glosses = json.loads(sys.argv[2])

learning_rate = 0.0003
epochs = 10
batch_size = 10
# The comparable token whose input and output embeddings are compared before and after fine-tuning.
# It should be a token that appears often in the fine-tuning dataset.
comparable_token = " minimalist"

# Dynamic import of the dataset
module_name = f"dataset_{bliss_id}_{'_'.join([gloss.replace(' ', '') for gloss in glosses])}"

try:
    user_dataset = importlib.import_module(f"{module_name}")
except ModuleNotFoundError:
    print(f"Error: Module '{module_name}' not found.")
    sys.exit(1)

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# save_model_dir = os.path.expanduser("~") + "/Development/LLMs/fine_tune_wool_shop_token"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
save_model_dir = os.path.expanduser("~") + f"/projects/ctb-whkchun/s2_bliss_LLMs/BLISS_{bliss_id}"
new_token = f"[BLISS_{bliss_id}]"
dtype = torch.bfloat16

# Load datasets
training_positive_context_sentences = user_dataset.training_positive_context_sentences
training_negative_context_sentences = user_dataset.training_negative_context_sentences
fine_tuning_sentences = user_dataset.fine_tuning_sentences
validation_sentences = user_dataset.validation_sentences

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print("Model loaded with 4-bit quantization. Tokenizer loaded.")

target_tokens = [tokenizer.tokenize(token)[0] for token in glosses]
target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
print(f"Target tokens: {target_tokens}; Target token IDs: {target_token_ids}")

# Calculate the output embedding for the new token
new_token_output_embedding_before = calc_output_embedding(model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids, dtype)
print("Calculated output embedding for the new token")

# Add the new token to the model with the new output embedding. The optimization of the input embedding needs it.
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"New token '{new_token}' added to the model with ID: {new_token_id}")

with torch.no_grad():
    model.get_output_embeddings().weight[new_token_id] = new_token_output_embedding_before.clone()
print(f"Output embedding of the new token '{new_token}' set to the calculated output embedding")

# Track the total running time of this script
start_time = time.time()

new_token_input_embedding_before = get_average_input_embeddings(model, tokenizer, glosses)
# Reset the input embedding of the new token to the initial token
model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()
print(f"Input embedding of the new token '{new_token}' set to the average input embedding of all constituent tokens of '{glosses}'")

# Use the comparable token to check if existing embeddings are changed by the fine-tuning
comparable_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comparable_token))[0]
print(f"Token ID of the comparable token '{comparable_token}': {comparable_token_id}\n")
comparable_token_input_embedding_before = model.get_input_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_output_embedding_before = model.get_output_embeddings().weight.data[comparable_token_id].float().detach()

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
    Dataset.from_dict({"text": fine_tuning_sentences})
    .map(tokenize_func, batched=True)
)

validation_dataset = (
    Dataset.from_dict({"text": validation_sentences})
    .map(tokenize_func, batched=True)
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=save_model_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=3,
    gradient_checkpointing=True,  # Reduces memory usage
    num_train_epochs=epochs,
    learning_rate=learning_rate,
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

# Load the best model to make sure the evaluation is done on the best model
best_model_path = trainer.state.best_model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
# Load base model and resize embeddings as the fine-tuned adapter has one more token than the base model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=dtype
)
model.resize_token_embeddings(len(tokenizer))
# Load PEFT adapter from checkpoint
model = PeftModel.from_pretrained(
    model,
    best_model_path,
    is_trainable=False  # For inference
)
# Merge adapter weights with base model
model = model.merge_and_unload()
print(f"Loaded best model from step {trainer.state.best_global_step}. Its best eval_loss = {trainer.state.best_metric}")

end_time_load_best_model = time.time()
elapsed_time = end_time_load_best_model - end_time_fine_tuning
print(f"Loading best model time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Save Model
model.save_pretrained(save_model_dir)
tokenizer.save_pretrained(save_model_dir)

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

comparable_token_input_embedding_after = model.get_input_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_input_embedding_similarity = torch.nn.functional.cosine_similarity(comparable_token_input_embedding_before.to(model.device), comparable_token_input_embedding_after.to(model.device), dim=0)
print(f"Cosine Similarity of {comparable_token} input embedding before and after: {comparable_token_input_embedding_similarity:.4f}")
distance = torch.norm(comparable_token_input_embedding_before.to(model.device) - comparable_token_input_embedding_after.to(model.device), p=2)
print(f"Euclidean Distance of {comparable_token} input embedding before and after: {distance.item():.4f}")

comparable_token_output_embedding_after = model.get_output_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_output_embedding_similarity = torch.nn.functional.cosine_similarity(comparable_token_output_embedding_before.to(model.device), comparable_token_output_embedding_after.to(model.device), dim=0)
print(f"Cosine Similarity of the {comparable_token} output embedding before and after: {comparable_token_output_embedding_similarity:.4f}")
distance = torch.norm(comparable_token_output_embedding_before.to(model.device) - comparable_token_output_embedding_after.to(model.device), p=2)
print(f"Euclidean Distance of {comparable_token} output embedding before and after: {distance.item():.4f}")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
testing_positive_context_sentences = user_dataset.testing_positive_context_sentences
testing_negative_context_sentences = user_dataset.testing_negative_context_sentences
testing_text_generation_prompts = user_dataset.testing_text_generation_prompts

evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, glosses
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
