import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_dir = "/home/cindyli/projects/ctb-whkchun/s2_bliss/results-finetune-7b-hf-3epochs/checkpoint-975"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


# Evaluation
# Inference. Generated texts by giving an instruction
def generate_text_with_instruction(instruction, input, model, tokenizer, temperature=0.7):
    input_key = "### Input:\n"
    response_key = "### Response:\n"
    prompt = f"{instruction}{input_key}{input}\n\n{response_key}\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    print(f"Instruction: {instruction}\n")
    print(f"Prompt: {input}\n")
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=temperature)
    print(f"Generated instruction (temprature {temperature}): {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n\n")


def generate_text_with_prompt(prompt, model, tokenizer, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    print(f"Prompt: {prompt}\n")
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=temperature)
    print(f"Generated instruction (temprature {temperature}): {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n\n")


# Test with exactly same instructions used in fine-tuning
instruction = "### Instruction: \nConvert the input English sentence to a Bliss sentence.\n\n"
input = "I am a programmer."
generate_text_with_instruction(instruction, input, model, tokenizer)

input = "Joe will explore the picturesque landscapes of a charming countryside village tomorrow."
generate_text_with_instruction(instruction, input, model, tokenizer)

input = "I had the pleasure of watching a captivating movie that thoroughly engaged my senses and emotions, providing a delightful escape into the realm of cinematic storytelling."
generate_text_with_instruction(instruction, input, model, tokenizer)

instruction = "### Instruction: \nConvert the input Bliss sentence to a English sentence.\n\n"
input = "past:The girl run in the park."
generate_text_with_instruction(instruction, input, model, tokenizer)

input = "future:month next, I embark on an journey exciting to explore the cultures vibrant and landscapes breathtaking of Southeast Asia."
generate_text_with_instruction(instruction, input, model, tokenizer)

# Test with random propmts
# Two prompts below is copied from the dataset
prompt = "Convert this sentence to a Bliss sentence: He rode his skateboard at the skate park yesterday.\n"
generate_text_with_prompt(prompt, model, tokenizer)

prompt = "Convert this Bliss sentence to an English sentence: present:They play merrily engage board by games cozy fireplace cozy evening cozy.\n"
generate_text_with_prompt(prompt, model, tokenizer)

prompt = "Write a Bliss sentence of a greeting.\n"
generate_text_with_prompt(prompt, model, tokenizer)

prompt = "Convert this sentence to a Bliss sentence: The Moon takes about one month to orbit Earth.\n"
generate_text_with_prompt(prompt, model, tokenizer)

prompt = "Convert this sentence to a Bliss sentence: He studied hard because he wanted to go to medical school as he suffered from arthritis.\n"
generate_text_with_prompt(prompt, model, tokenizer)

prompt = "Convert this Bliss sentence to an English sentence: past:he ride excitedly bike shiny new down road day next.\n"
generate_text_with_prompt(prompt, model, tokenizer)
