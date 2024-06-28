# Copyright (c) 2024, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.globals import set_debug
# set_debug(True)

# Define the Ollama model to use
model = "llama3"

# Define the number of the most recent chats to be passed in as the most recent chats.
# The summary of chats before the most recent will be passed in as another context element.
num_of_recent_chat = 1

# Telegraphic reply to be translated
message_to_convert = "she love cooking like share recipes"

# Chat history
chat_history = [
    "John: Have you heard about the new Italian restaurant downtown?",
    "Elaine: Yes, I did! Sarah mentioned it to me yesterday. She said the pasta there is amazing.",
    "John: I was thinking of going there this weekend. Want to join?",
    "Elaine: That sounds great! Maybe we can invite Sarah too.",
    "John: Good idea. By the way, did you catch the latest episode of that mystery series we were discussing last week?",
    "Elaine: Oh, the one with the detective in New York? Yes, I watched it last night. It was so intense!",
    "John: I know, right? I didn't expect that plot twist at the end. Do you think Sarah has seen it yet?",
    "Elaine: I'm not sure. She was pretty busy with work the last time we talked. We should ask her when we see her at the restaurant.",
    "John: Definitely. Speaking of Sarah, did she tell you about her trip to Italy next month?",
    "Elaine: Yes, she did. She's so excited about it! She's planning to visit a lot of historical sites.",
    "John: I bet she'll have a great time. Maybe she can bring back some authentic Italian recipes for us to try.",
]
recent_chat_array = []
earlier_chat_array = []

# 1. Instantiate the chat model and split the chat history
llm = ChatOllama(model=model)

if (len(chat_history) > num_of_recent_chat):
    recent_chat_array = chat_history[-num_of_recent_chat:]
    earlier_chat_array = chat_history[:-num_of_recent_chat]
else:
    recent_chat_array = chat_history
    earlier_chat_array = []

# 2. Summarize earlier chat
if (len(earlier_chat_array) > 0):
    summarizer_prompt = ChatPromptTemplate.from_template("Summarize the following chat history. Provide only the summary, without any additional comments or context. \nChat history: {chat_history}")
    chain = summarizer_prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "chat_history": "\n".join(earlier_chat_array)
    })
print("====== Summary ======")
print(f"{summary}\n")

# 3. concetenate recent chat into a string
recent_chat_string = "\n".join(recent_chat_array)
print("====== Recent Chat ======")
print(f"{recent_chat_string}\n")

# Create prompt template
prompt_template_with_context = """
### Elaine prefers to talk using telegraphic messages. Help to convert Elaine's reply to a chat into full sentences in first-person. Only respond with the converted full sentences.

### This is the chat summary:

{summary}

### This is the most recent chat between Elaine and others:

{recent_chat}

### This is Elaine's most recent response to continue the chat. Please convert:
{message_to_convert}
"""

prompt = ChatPromptTemplate.from_template(prompt_template_with_context)

# using LangChain Expressive Language (LCEL) chain syntax
chain = prompt | llm | StrOutputParser()

print("====== Response without chat history ======")

print(chain.invoke({
    "summary": "",
    "recent_chat": recent_chat_string,
    "message_to_convert": message_to_convert
}) + "\n")

print("====== Response with chat history ======")

print(chain.invoke({
    "summary": summary,
    "recent_chat": recent_chat_string,
    "message_to_convert": message_to_convert
}) + "\n")
