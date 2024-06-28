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

# Telegraphic reply to be translated
message_to_convert = "she love cooking like share recipes"

# Conversation history
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

# Instantiate the chat model and split the conversation history
llm = ChatOllama(model=model)

# Create prompt template
prompt_template_with_context = """
Elaine prefers to talk using telegraphic messages.
Given a chat history and Elaine's latest response which
might reference context in the chat history, convert
Elaine's response to full sentences. Only respond with
converted full sentences.

Chat history:
{chat_history}

Elaine's response:
{message_to_convert}
"""

prompt = ChatPromptTemplate.from_template(prompt_template_with_context)

# using LangChain Expressive Language (LCEL) chain syntax
chain = prompt | llm | StrOutputParser()

print("====== Response without chat history ======")

print(chain.invoke({
    "chat_history": "",
    "message_to_convert": message_to_convert
}) + "\n")

print("====== Response with chat history ======")

print(chain.invoke({
    "chat_history": "\n".join(chat_history),
    "message_to_convert": message_to_convert
}) + "\n")
