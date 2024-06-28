# Copyright (c) 2023-2024, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

import sys
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# A utility function that prints the script usage then exit
def printUsageThenExit():
    print("Usage: python rag.py <sentence_transformer_model_directory>")
    sys.exit()


# Read the path to the sentence transformer model
if len(sys.argv) != 2:
    printUsageThenExit()
else:
    sentence_transformer_dir = sys.argv[1]
    if not os.path.isdir(sentence_transformer_dir):
        printUsageThenExit()

# The location of the user document
user_doc = "./data/user_doc.txt"

loader = TextLoader(user_doc)
documents = loader.load()
# print(f"Loaded documents (first 2 rows):\n{documents[:2]}\n\n")

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
splitted_docs = text_splitter.split_documents(documents)
# print(f"Splitted documents (first 2 rows):\n{splitted_docs[:2]}\n\n")

# Instantiate the embedding class
embedding_func = HuggingFaceEmbeddings(model_name=sentence_transformer_dir)

# Load into the vector database
vectordb = FAISS.from_documents(splitted_docs, embedding_func)

# Create a vector store retriever
retriever = vectordb.as_retriever()

# query the vector db to test
queries = [
    "Roy nephew",
    "high school"]

for query in queries:
    results = retriever.invoke(query)
    print(f"====== Test: Similarity search for \"{query}\" ======\n{results[0].page_content}\n\n")

# Create prompt template
prompt_template_with_context = """
### [INST] Help to convert Elaine's telegraphic input in the conversation to full sentences in first-person. Only respond with the converted full sentences. Here is context to help:

{context}

### Conversation:
{chat} [/INST]
 """

llm = ChatOllama(model="llama3", system="Elaine is an AAC user who expresses herself telegraphically. She is now in a meeting with Jutta. Below is the conversation in the meeting. Please help to convert what Elaine said to first-person sentences. Only respond with converted sentences.")
prompt = ChatPromptTemplate.from_template(prompt_template_with_context)

elaine_reply = "Roy nephew"
full_chat = f"Jutta: Elaine, who would you like to invite to your birthday party?\n Elaine: {elaine_reply}."

# using LangChain Expressive Language (LCEL) chain syntax
chain = prompt | llm | StrOutputParser()

print("====== Response without RAG ======")

print(chain.invoke({
    "context": "",
    "chat": full_chat
}) + "\n")

print("====== Response with RAG ======")

print(chain.invoke({
    "context": retriever.invoke(elaine_reply),
    "chat": full_chat
}) + "\n")
