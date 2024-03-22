import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
# from peft import PeftModel
from trl import SFTTrainer

# The local directory with the model and the tokenizer
model_dir = "/home/cindyli/projects/ctb-whkchun/s2_bliss_LLMs/Llama-2-7b-hf"

# The instruction dataset to use
dataset_name = "/home/cindyli/llama2/finetune/bliss.json"

# Output directory where the model checkpoints will be stored
output_dir = "/home/cindyli/projects/ctb-whkchun/s2_bliss/results-finetune-7b-hf"

# Fine-tuned model name
new_model = "llama-2-7b-hf-bliss"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 4096

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


# Create a formatted prompt template for an entry in the dataset
# The "direction" parameter accepts either "EnglishToBliss" or "BlissToEnglish"
def format_prompt(sample, direction="EnglishToBliss"):
    # Initialize static strings for the prompt template
    instruction = f"### Instruction: \nConvert the input {'English' if direction=='EnglishToBliss' else 'Bliss'} sentence to a {'Bliss' if direction=='EnglishToBliss' else 'English'} sentence.\n\n"
    input_key = "### Input:\n"
    response_key = "### Response:\n"

    if direction == "EnglishToBliss":
        # Format the sample for English to Bliss conversion
        sample["text"] = f"{instruction}{input_key}{sample['original']}\n\n{response_key}{sample['bliss']}\n"
    elif direction == "BlissToEnglish":
        # Format the sample for Bliss to English conversion
        sample["text"] = f"{instruction}{input_key}{sample['bliss']}\n\n{response_key}{sample['original']}\n"

    return sample


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # Fix weird overflow issue with fp16 training

# Preprocess dataset
print("Preprocessing dataset...")
orig_dataset = load_dataset("json", data_files=dataset_name, split="train")
print(f"Number of prompts: {len(orig_dataset)}")
print(f"Column names are: {orig_dataset.column_names}")

# Create two datasets for English to Bliss and Bliss to English
english_to_bliss_dataset = orig_dataset.map(lambda x: format_prompt(x, direction="EnglishToBliss"))
bliss_to_english_dataset = orig_dataset.map(lambda x: format_prompt(x, direction="BlissToEnglish"))

# Combine the two datasets
dataset = concatenate_datasets([english_to_bliss_dataset, bliss_to_english_dataset])

print(dataset)
print(f"First record: {dataset[0]}")
print(f"Last record: {dataset[-1]}")
print("Done with preprocessing dataset.\n\nStart fine-tuning...")

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=f"{output_dir}-{num_train_epochs}epochs",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(f"{new_model}-{num_train_epochs}epochs")

print("Done with the fine-tuning.")

# Evaluate the new model


# Inference
def generate_text(instruction, input, model, tokenizer):
    input_key = "### Input:\n"
    response_key = "### Response:\n"
    prompt = f"{instruction}{input_key}{input}\n\n{response_key}\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.9)
    print(f"Instruction: {instruction}\n")
    print(f"Prompt: {input}\n")
    print(f"Generated instruction: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n\n")


# Word predictions
def predict_words(prompt, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    predictions = pipe(prompt, max_length=20, num_return_sequences=3)

    # write results into the result file
    print(f"## Prompt: {prompt}\n## Predictions:\n")
    for prediction in predictions:
        print(f"- {prediction['generated_text']}\n")
    print("\n\n")


print("1. Inference\n")
instruction = "### Instruction: \nConvert the input English sentence to a Bliss sentence.\n\n"
input = "I am a programmer."
generate_text(instruction, input, model, tokenizer)

input = "Joe will explore the picturesque landscapes of a charming countryside village tomorrow."
generate_text(instruction, input, model, tokenizer)

input = "I had the pleasure of watching a captivating movie that thoroughly engaged my senses and emotions, providing a delightful escape into the realm of cinematic storytelling."
generate_text(instruction, input, model, tokenizer)

instruction = "### Instruction: \nConvert the input Bliss sentence to a English sentence.\n\n"
input = "past:The girl run in the park."
generate_text(instruction, input, model, tokenizer)

input = "future:month next, I embark on an journey exciting to explore the cultures vibrant and landscapes breathtaking of Southeast Asia."
generate_text(instruction, input, model, tokenizer)

print("2. Word Prediction\n\n")
prompt = "present: Joe be in hospital. He"
predict_words(prompt, model, tokenizer)

prompt = "Tomorrow will be a beautiful day. Running"
predict_words(prompt, model, tokenizer)

# Empty VRAM
del model
del trainer
