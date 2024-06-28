# Baby Bliss Bot

An exploratory research project to generate new Bliss vocabulary using machine learning techniques.

[The Bliss language](https://www.blissymbolics.org/) is an Augmentative and Alternative Communication (AAC) language
used by individuals with severe speech and physical impairments around the world, but also by others for language
learning and support, or just for the fascination and joy of this unique language representation. It is a semantic
graphical language that is currently composed of more than 5000 authorized symbols - Bliss-characters and Bliss-words.
It is a generative language that allows its users to create new Bliss-words as needed.

We are exploring the generation of new Bliss vocabulary using emerging AI techniques, including Large Language Models
(LLM), OCR, and other models for text generation and completion.

## Local Installation

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
  * Version 3.9+. On Mac, Homebrew is the easiest way to install.

### Clone the Repository

* Clone the project from GitHub. [Create a fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
with your GitHub account, then run the following in your command line (make sure to replace `your-username` with
your username):

```bash
git clone https://github.com/your-username/baby-bliss-bot
cd baby-bliss-bot
```

### Create/Activitate Virtual Environment
Always activate and use the python virtual environment to maintain an isolated environment for project's dependencies.

* [Create the virtual environment](https://docs.python.org/3/library/venv.html)
  (one time setup): 
  - `python -m venv .venv` 

* Activate (every command-line session):
  - Windows: `.\.venv\Scripts\activate`
  - Mac/Linux: `source .venv/bin/activate`

### Install Python Dependencies

Run in the baby-bliss-bot directory:
* `pip install -r requirements.txt`

## Linting

Run the following command to lint all python scripts:

* `flake8`

## Model Experiments

We performed experiments with a number of existing models listed below to understand how useful they are in helping
with generating new Bliss symbols etc.

### Llama2

**Conclusion**: useful

See the [Llama2FineTuning.md](./docs/Llama2FineTuning.md) in the [documentation](./docs) folder for details
on how to fine tune, evaluation results and the conclusion about how useful it is.

### StyleGAN3

**Conclusion**: not useful

See the [TrainStyleGAN3Model.md](./docs/TrainStyleGAN3Model.md) in the [documentation](./docs) folder for details
on how to train this model, training results and the conclusion about how useful it is.

### StyleGAN2-ADA

**Conclusion**: shows promise

See the [StyleGAN2-ADATraining.md](./docs/StyleGAN2-ADATraining.md) in the [documentation](./docs) folder for details
on how to train this model and training results.

### Texture Inversion

**Conclusion**: not useful 

See the [Texture Inversion documentation](./notebooks/README.md) for details.

## Preserving Information

### RAG (Retrieval-augmented generation)

**Conclusion**: useful

RAG (Retrieval-augmented generation) technique is explored to resolve ambiguities by retrieving relevant contextual
information from external sources, enabling the language model to generate more accurate and reliable responses.

See [RAG.md](./docs/RAG.md) for more details.

### Reflection over Chat History

**Conclusion**: useful

When users have a back-and-forth conversation, the application requires a form of "memory" to retain and incorporate past interactions into its current processing. Two methods are explored to achieve this:

1. Summarizing the chat history and providing it as contextual input.
2. Using prompt engineering to instruct the language model to consider the past conversation.

The second method, prompt engineering, yields more desired responses than summarizing chat history.

See [ReflectChatHistory.md](./docs/RAG.md) for more details.

## Notebooks

[`/notebooks`](./notebooks/) directory contains all notebooks used for training or fine-tuning various models.
Each notebook usually comes with a accompanying `dockerfile.yml` to elaborate the environment that the notebook was
running in.

## Jobs
[`/jobs`](./jobs/) directory contains all jobs and scripts used for training or fine-tuning various models, as well
as other explorations with RAG (Retrieval-augmented generation) and preserving chat history.

## Utility Scripts

All utility functions are in the [`utils`](./utils) directory. 

See [README.md](./utils/README.md) in the [`utils`](./utils) directory for details.
