import torch
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline
)

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma

# The location of the user document
user_files = ["/home/cindyli/mistral/RAG/data/user_doc.txt"]

# The local directory with the model and the tokenizer
model_dir = "/home/cindyli/projects/ctb-whkchun/s2_bliss_LLMs/Mistral-7B-Instruct-v0.2"

# The local directory with the sentence transformer
sentence_transformer_dir = "/home/cindyli/projects/ctb-whkchun/s2_bliss_LLMs/all-MiniLM-L6-v2"

# DB directory
# db_directory = "chroma_local_db"

# Output directory where the model checkpoints will be stored
output_dir = "~/projects/ctb-whkchun/s2_bliss/results-rag-mistral"

################################################################################
# Load user documents, split into small pieces and create embeddings for them
################################################################################

# Load user documents
loader = UnstructuredFileLoader(user_files, mode="elements")
raw_documents = loader.load()
print(f"Loaded documents (first 10 rows):\n{raw_documents[:10]}")

# split into small pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
splitted_documents = text_splitter.split_documents(raw_documents)
print(f"Splitted documents (first 10 rows):\n{splitted_documents[:10]}")

# Instantiate the embedding class
embedding_func = HuggingFaceEmbeddings(model_name=sentence_transformer_dir)

# Load into Chroma, the vector database
vectordb = Chroma.from_documents(splitted_documents, embedding_func)

# query the vector db to test
queries = [
    "What's the name of Elaine's nephew?",
    "What schools did Elaine attend?"]

for query in queries:
    result = vectordb.similarity_search(query)
    print(f"Similarity search in Chroma returns for {query}:\n{result[0].page_content}")

# Create a vector store retriever
retriever = vectordb.as_retriever()

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
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

#################################################################
# Load tokenizer and model
#################################################################

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # Fix weird overflow issue with fp16 training

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    quantization_config=bnb_config
)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

#################################################################
# Create RAG chain
#################################################################

# Create prompt template
prompt_template = """
### [INST] Instruction: Elaine is an AAC user who expresses herself telegraphically. She is now in a conversation with Jutta. Below is the conversation in the meeting. Please help to convert what Elaine said to first-person sentences. Only respond with converted sentences. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | llm_chain)

rag_chain.invoke("Jutta:  Who would you like to invite to your birthday party?\nElaine: Roy, nephew.")
