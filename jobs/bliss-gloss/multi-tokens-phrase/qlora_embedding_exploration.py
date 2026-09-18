# This script explores when adding a new token into the vocabulary of a pretrained model to represent
# a certain meaning, how the initializing of its input embedding and output embedding before the QLoRA
# fine-tuning affects the model's performance. It also finds out the embedding of the new token is
# considered as a trainable parameter in the QLoRA fine-tuning process, so only fine-tune the attention
# layers is enough because the embedding of the new token will be fine-tuned as well.

# This script tested 5 different cases for initializing the input embedding and the output embedding
# of the new token:
# 1. Initialize input embedding to an optimized input embedding & use calcuated output embedding.
# 2. Initialize input embedding to the averaged input embeddings of all composing tokens & use calcuated output embedding.
# 3. Initialize input embedding to the input embedding of the last token in the gloss & use calcuated output embedding.
# 4. Initialize input embedding to a random embedding & use calcuated output embedding.
# 5. Initialize both input embedding and output embedding to random embeddings.

# Conclusions:
# Calculated output embedding works better than fine-tuned random output embedding
# Initialize input embedding with a relevant initial value works better than a random input embedding
# The embedding of the new token is considered as a trainable parameter in the QLoRA fine-tuning process,
# so only fine-tune the attention layers is enough because the embedding of the new token will be fine-tuned
# as well.

# python qlora_embedding_exploration.py <learning_rate> <epochs> <batch_size>
# python qlora_embedding_exploration.py 0.0003 10 10

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

# The path to the best model is saved in the Trainer state, and can be accessed using: trainer.state.best_model_checkpoint
# This path is also saved in "output_dir/checkpoint-<step>/trainer_state.json" -> "best_model_checkpoint" path

import os
import sys
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import dataset_29111_wool_shop as user_dataset
from utils import evaluate_new_token, get_average_input_embeddings  # noqa: E402
# from data import dataset_29111_wool_shop as user_dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import evaluate_new_token, get_average_input_embeddings  # noqa: E402

if len(sys.argv) != 4:
    print("Usage: python qlora_embedding_exploration.py <learning_rate> <epochs> <batch_size>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

case_name = "Initialize Input Embedding to ' shop', Calculated Output Embedding, Fine-tune only attention layers"
save_model_dir_suffix = "/random_attention_layers_and_only_new_input_embedding"

# Embedding flags
# When all flags are False, the original input embedding in the loaded model will be used: optimized input embedding & calculated output embedding

# When True, the input embedding of the new token will be initialized to the averaged input embeddings of all composing tokens.
AVERAGE_INPUT_EMBEDDING = False
# When True, the input embedding of the new token will be initialized to the input embedding of a given token.
RESET_INPUT_EMBEDDING_TO_INITIAL_TOKEN = True
# When True, the input embedding of the new token will be initialized to a random embedding.
RANDOMIZE_INPUT_EMBEDDING = False
# When True, both input embedding and output embedding of the new token will be initialized to random embeddings.
RANDOMIZE_BOTH_EMBEDDINGS = False

# Fine-tuning flags

# When all flags are False, will fine-tune both attention layers and the input embedding layer.
# When True, only fine tune attention layers.
FINE_TUNE_ONLY_ATTENTION_LAYERS = True
# When True, only fine tune the new token's input embedding layer and attention layers.
FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS = False

print("Parameters:")
print(f"- learning_rate: {learning_rate}")
print(f"- epochs: {epochs}")
print(f"- batch_size: {batch_size}")
print(f"- AVERAGE_INPUT_EMBEDDING: {AVERAGE_INPUT_EMBEDDING}")
print(f"- RESET_INPUT_EMBEDDING_TO_INITIAL_TOKEN: {RESET_INPUT_EMBEDDING_TO_INITIAL_TOKEN}")
print(f"- RANDOMIZE_INPUT_EMBEDDING: {RANDOMIZE_INPUT_EMBEDDING}")
print(f"- RANDOMIZE_BOTH_EMBEDDINGS: {RANDOMIZE_BOTH_EMBEDDINGS}")
print(f"- FINE_TUNE_ONLY_ATTENTION_LAYERS: {FINE_TUNE_ONLY_ATTENTION_LAYERS}")
print(f"- FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS: {FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS}\n")

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# save_model_dir = os.path.expanduser("~") + "/Development/LLMs/fine_tune_wool_shop_token"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding_lr0.001/epochs1500"
save_model_dir = os.path.expanduser("~") + f"/projects/ctb-whkchun/s2_bliss_LLMs/qlora_fine_tune_BLISS_29111/{save_model_dir_suffix}"
new_token = "[BLISS_29111]"   # Token for "wool shop"

# Load datasets
fine_tuning_sentences = user_dataset.fine_tuning_sentences
validation_sentences = user_dataset.validation_sentences

# Track the total running time of this script
start_time = time.time()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Case: {case_name}\n")

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

new_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_token))[0]
print(f"Token ID of the new token '{new_token}': {new_token_id}")

comparable_token = " knitting"  # The token to find if its embeddings are changed by the fine-tuning
comparable_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comparable_token))[0]
print(f"Token ID of the comparable token '{comparable_token}': {comparable_token_id}")

optimized_input_embedding = model.get_input_embeddings().weight.data[new_token_id].float().detach()
calc_output_embedding = model.get_output_embeddings().weight.data[new_token_id].float().detach()
new_token_input_embedding_before = optimized_input_embedding
new_token_output_embedding_before = calc_output_embedding

comparable_token_input_embedding_before = model.get_input_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_output_embedding_before = model.get_output_embeddings().weight.data[comparable_token_id].float().detach()

if AVERAGE_INPUT_EMBEDDING:
    print(f"Resetting input embedding of {new_token} to the average input embedding of all constituent tokens")
    initial_input_embedding = get_average_input_embeddings(model, tokenizer, [" wool shop"])
    # Reset the input embedding of the new token to the initial token
    model.get_input_embeddings().weight.data[new_token_id] = initial_input_embedding.clone()
    new_token_input_embedding_before = initial_input_embedding

if RESET_INPUT_EMBEDDING_TO_INITIAL_TOKEN:
    initial_token = " shop"  # Initial token to reset the embeddings for the new token based on the flags
    initial_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(initial_token))[0]
    print(f"Token ID of the initial token '{initial_token}': {initial_token_id}")

    print(f"Resetting input embedding of {new_token} to be the same as the initial token '{initial_token}'")
    new_token_input_embedding_before = model.get_input_embeddings().weight.data[initial_token_id]
    # Reset the input embedding of the new token to the initial token
    model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()

if RANDOMIZE_INPUT_EMBEDDING:
    print(f"Resetting input embedding of {new_token} to a random embedding")
    # Randomize the input embedding of the new token
    new_token_input_embedding_before = torch.randn(model.config.hidden_size, device=model.device)
    model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()

if RANDOMIZE_BOTH_EMBEDDINGS:
    print(f"Resetting both input embedding and output embedding of {new_token} to random embeddings")
    # Randomize the input embedding of the new token
    new_token_input_embedding_before = torch.randn(model.config.hidden_size, device=model.device)
    model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before.clone()

    new_token_output_embedding_before = torch.randn(model.config.hidden_size, device=model.device)
    model.get_output_embeddings().weight.data[new_token_id] = new_token_output_embedding_before.clone()

# Preprocess the quantized model for QLoRA Training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
target_modules = (["q_proj", "k_proj", "v_proj", "o_proj"] if FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS or FINE_TUNE_ONLY_ATTENTION_LAYERS else ["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens"])
print(f"Target modules for LoRA: {target_modules}")

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)
model = get_peft_model(model, peft_config)

if FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS:
    # Unfreeze the new token's embedding in the base model
    base_model = model.get_base_model()
    embeddings = base_model.get_input_embeddings().weight
    embeddings.requires_grad = True   # Unfreeze the entire embedding layer


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
    weight_decay=0.0,    # prevent decay on embeddings
    bf16=False,    # Prevent precision-related issue when comparing embeddings before and after fine-tuning
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


# A callback to zero gradients except for the new token
class EmbedGradCallback(TrainerCallback):
    def on_backward(self, args, state, control, **kwargs):
        print("=== Gradient Masking Callback ===")
        embeddings = kwargs['model'].get_base_model().get_input_embeddings().weight
        if embeddings.grad is not None:
            # The debug print to confirm gradients are only applied to the new token
            print("Max gradient for non-target tokens:",
                  torch.max(embeddings.grad[torch.arange(embeddings.size(0)) != new_token_id]).item())

            # Keep only the new token's gradient
            grad_mask = torch.zeros_like(embeddings.grad)
            grad_mask[new_token_id] = embeddings.grad[new_token_id]
            embeddings.grad.copy_(grad_mask)


# Create Trainer
trainer_callbacks = ([EmbedGradCallback()] if FINE_TUNE_ONLY_NEW_TOKEN_INPUT_EMBEDDING_AND_ATTENTION_LAYERS else [])
print(f"Trainer callbacks: {trainer_callbacks}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    callbacks=trainer_callbacks,
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
print(f"Fine-tuning time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

best_model_path = trainer.state.best_model_checkpoint
model = AutoModelForCausalLM.from_pretrained(best_model_path)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
print(f"Loaded best model from step {trainer.state.best_global_step}. Its best eval_loss = {trainer.state.best_metric}")

end_time_load_best_model = time.time()
elapsed_time = end_time_load_best_model - end_time_fine_tuning
print(f"Loading best model time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Save Model
model.save_pretrained(save_model_dir, save_embedding_layers=True)
tokenizer.save_pretrained(save_model_dir)

# Compare the embeddings of tokens before and after fine-tuning
new_token_input_embedding_after = model.get_input_embeddings().weight.data[new_token_id].float().detach()
input_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_input_embedding_before.to(model.device), new_token_input_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {new_token} input embedding before and after: {input_embedding_similarity:.4f}")

new_token_output_embedding_after = model.get_output_embeddings().weight.data[new_token_id].float().detach()
output_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_output_embedding_before.to(model.device), new_token_output_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {new_token} output embedding before and after: {output_embedding_similarity:.4f}\n")

input_embedding_similarity = torch.nn.functional.cosine_similarity(optimized_input_embedding.to(model.device), new_token_input_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {new_token} input embedding btw optimized and after: {input_embedding_similarity:.4f}")

output_embedding_similarity = torch.nn.functional.cosine_similarity(calc_output_embedding.to(model.device), new_token_output_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {new_token} output embedding btw calculated and after: {output_embedding_similarity:.4f}\n")

comparable_token_input_embedding_after = model.get_input_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_input_embedding_similarity = torch.nn.functional.cosine_similarity(comparable_token_input_embedding_before.to(model.device), comparable_token_input_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {comparable_token} input embedding before and after: {comparable_token_input_embedding_similarity:.4f}")

comparable_token_output_embedding_after = model.get_output_embeddings().weight.data[comparable_token_id].float().detach()
comparable_token_output_embedding_similarity = torch.nn.functional.cosine_similarity(comparable_token_output_embedding_before.to(model.device), comparable_token_output_embedding_after.to(model.device), dim=0)
print(f"Similarity of the {comparable_token} output embedding before and after: {comparable_token_output_embedding_similarity:.4f}")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
training_positive_context_sentences = user_dataset.training_positive_context_sentences
training_negative_context_sentences = user_dataset.training_negative_context_sentences
testing_positive_context_sentences = user_dataset.testing_positive_context_sentences
testing_negative_context_sentences = user_dataset.testing_negative_context_sentences
testing_text_generation_prompts = user_dataset.testing_text_generation_prompts

target_tokens = [" wool", " yarn"]
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, " wool shop"
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# # Clean up
# del model, trainer
# gc.collect()
# torch.cuda.empty_cache()
