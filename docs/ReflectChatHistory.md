# Reflection over Chat History

When users have a back-and-forth conversation, the application requires a form of "memory" to retain and incorporate
past interactions into its current processing. Two methods are explored to achieve this:

1. Summarizing the chat history and providing it as contextual input.
2. Using prompt engineering to instruct the language model to consider the past conversation.

The second method, prompt engineering, yields more desired responses than summarizing chat history.

The scripts for this experiment is located in the `jobs/RAG` directory.

## Method 1: Summarizing the Chat History

### Steps

1. Summarize the past conversation and include it in the prompt as contextual information.
2. Include a specified number of the most recent conversation exchanges in the prompt for additional context.
3. Instruct the language model to convert the telegraphic replies from the AAC user into full sentences to continue
the conversation.

### Result

The conversion process struggles to effectively utilize the provided summary, often resulting in inaccurate full
sentences.

### Scripts

* `requirements.txt`: Lists the Python dependencies needed to set up the environment.
* `chat_history_with_summary.py`: Implements the steps described above and displays the output.

## Method 2: Using Prompt Engineering

### Steps

1. Include the past conversation in the prompt as contextual information.
2. Instruct the language model to reference this context when converting the telegraphic replies from the AAC user
into full sentences to continue the conversation.

### Result

The converted sentences are more accurate and appropriate compared to those generated using Method 1.

### Scripts

* `requirements.txt`: Lists the Python dependencies needed to set up the environment.
* `chat_history_with_prompt.py`: Implements the steps described above and displays the output.

## Run Scripts Locally

### Prerequisites

* [Ollama](https://github.com/ollama/ollama) to run language models locally
  * Follow [README](https://github.com/ollama/ollama?tab=readme-ov-file#customize-a-model) to
  install and run Ollama on a local computer.
* If you are currently in a activated virtual environment, deactivate it.

### Create/Activitate Virtual Environment
* Go to the RAG scripts directory
  - `cd jobs/RAG`

* [Create the virtual environment](https://docs.python.org/3/library/venv.html)
  (one time setup): 
  - `python -m venv .venv` 

* Activate (every command-line session):
  - Windows: `.\.venv\Scripts\activate`
  - Mac/Linux: `source .venv/bin/activate`

* Install Python Dependencies (Only run once for the installation)
  - `pip install -r requirements.txt`

### Run Scripts
* Run `chat_history_with_summary.py` or `chat_history_with_prompt.py`
  - `python chat_history_with_summary.py` or `python chat_history_with_prompt.py`
  - The last two responses in the exectution result shows the language model's output
  with and without the contextual information.
