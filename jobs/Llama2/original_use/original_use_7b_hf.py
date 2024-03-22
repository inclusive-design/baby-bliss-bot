from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = "/home/cindyli/llama2/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
result_file = "/home/cindyli/llama2/original_use/result.txt"
output_file = open(result_file, "w")


# Text generation
def generate_text(prompt, output_file, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400)
    pipe_return = pipe(f"<s>[INST] {prompt} [/INST]")

    # write results into the result file
    output_file.write(f"## Text generation: \n{pipe_return[0]['generated_text']}\n\n")


# Word predictions
def predict_words(prompt, output_file, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    predictions = pipe(prompt, max_length=20, num_return_sequences=3)

    # write results into the result file
    output_file.write("## Word prediction: \n")
    for prediction in predictions:
        output_file.write(f"- {prediction['generated_text']}\n")
    output_file.write("\n")


# Generate inference
def get_inference(prompt, output_file, model, tokenizer):
    inference = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=300)[0])

    # write results into the result file
    output_file.write(f"## Inference: \n{inference}\n")


instruction = "Convert this English sentence to a structure in the Bliss language: "
generate_text(f"{instruction}The government says the measures are meant to address the cost of living and spur home building in the province while critics say the spending is reckless.",  output_file, model, tokenizer)
generate_text(f"{instruction}Today is a lovely sunny day.",  output_file, model, tokenizer)
generate_text(f"{instruction}Yesterday, I watched a captivating movie that thoroughly engaged my senses and emotions, providing a delightful escape into the realm of cinematic storytelling.",  output_file, model, tokenizer)
generate_text(f"{instruction}I will explore the picturesque landscapes of a charming countryside village.",  output_file, model, tokenizer)

# predict_words("Joe is feeling sick and in the hospital. He wants to",  output_file, model, tokenizer)
# get_inference("Joe wants to",  output_file, model, tokenizer)
