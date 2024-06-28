# Experiment with Retrieval-Augumented Generation (RAG)

Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of
generative AI models with facts fetched from external sources. This approach aims to address the
limitations of traditional language models, which may generate responses based solely on their
training data, potentially leading to factual errors or inconsistencies. Read 
[What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
for more information.

In a co-design session with an AAC (Augmentative and Alternative Communication) user, RAG can
be particularly useful. When the user expressed a desire to invite "Roy nephew" to her birthday
party, the ambiguity occurred as to whether "Roy" and "nephew" referred to the same person or
different individuals. Traditional language models might interpret this statement inconsistently,
sometimes treating "Roy" and "nephew" as the same person, and other times as separate persons.

RAG addresses this issue by leveraging external knowledge sources, such as documents or databases
containing relevant information about the user's family members and their relationships. By
retrieving and incorporating this contextual information into the language model's input, RAG
can disambiguate the user's intent and generate a more accurate response.

The RAG experiment is located in the `jobs/RAG` directory. It contains these scripts:

* `requirements.txt`: contains python dependencies for setting up the environment to run
the python script.
* `rag.py`: use RAG to address the "Roy nephew" issue described above.

## Run Scripts Locally

### Prerequisites

* If you are currently in a activated virtual environment, deactivate it.

* Install and start [Ollama](https://github.com/ollama/ollama) to run language models locally
  * Follow [README](https://github.com/ollama/ollama?tab=readme-ov-file#customize-a-model) to
  install and run Ollama on a local computer.

* Download a Sentence Transformer Model
  1. Select a Model
    - Choose a [sentence transformer model](https://huggingface.co/sentence-transformers) from Hugging Face.
  2. Download the Model
    - Download the selected model to a local directory. For example, to download the 
    [`all-MiniLM-L6-v2` model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), use the following
    command:
      ```sh
      git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
      ```
  3. Provide the Model Path
    - When running the `rag.py` script, provide the path to the directory of the downloaded model as a parameter.
  **Note:** Accessing a local sentence transformer model is much faster than accessing it via the
  `sentence-transformers` Python package.

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
* Run `rag.py` with a parameter providing the path to the directory of a sentence transformer model
  - `python rag.py ./all-MiniLM-L6-v2/`
  - The last two responses in the exectution result shows the language model's output
  with and without the use of RAG.
