# Baby Bliss Bot

An exploratory research project that uses generative AI to enhance communication for individuals who utilize
Augmentative and Alternative Communication (AAC) devices and rely on a minority language system.

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

### Create/Activate Virtual Environment

Always activate and use the python virtual environment to maintain an isolated environment for project's dependencies.

* [Create the virtual environment](https://docs.python.org/3/library/venv.html)
  (one time setup):
  * `python -m venv .venv`

* Activate (every command-line session):
  * Windows: `.\.venv\Scripts\activate`
  * Mac/Linux: `source .venv/bin/activate`

### Install Python Dependencies

Run in the baby-bliss-bot directory:

* `pip install -r requirements.txt`

## Linting

Run the following command to lint all python scripts:

* `flake8`

## Documentation

### Teaching Llama Model to Understand Bliss Grammatical Indicators

* **[Add Grammatical Indicator Symbols](./docs/AddGrammaticalIndicators.md)**  
  Describes three methods explored for integrating Bliss grammatical indicator symbols into a Llama model.

### Teaching Llama Model to Understand Bliss Meaning Symbols

* **[Integrating Bliss Meaning Symbols into the Model](./docs/IntegrateBlissMeaningSymbols.md)**  
  Describes the process of integrating Bliss Meaning Symbols into a LLaMA language model by introducing a dedicated token for each symbol. These symbols represent specific concepts (e.g., "tree", "house", etc.).

* **[Using Bliss Gloss as a Semantic Bridge](./docs/ExploreBlissGloss.md)**  
  Investigates the use of Bliss glosses to leverage the model’s pre-trained English knowledge. This approach explores mapping Bliss symbols to English phrases or concepts to enhance understanding.

* **[Disambiguating Synonyms](./docs/DisambiguateSynonyms.md)**  
  Explores techniques for resolving ambiguity in English words when mapping to Bliss symbols. Proposes computing output embeddings of Bliss tokens using training data to ensure conceptual precision.

* **[Use a Single Token for a Multi-token Gloss](./docs/InputEmbeddingForMultiTokenGloss.md)**  
  Explores methods for representing a multi-token gloss with a new single Bliss token. Compares several strategies for initializing its input embedding prior to fine-tuning and identifies the most effective approach.

* **[Adding Bliss Symbol Tokens](./docs/AddBlissSpecificTokens.md)**  
  Provides step-by-step instructions for introducing new Bliss symbol tokens into a LLaMA model. For a complete pipeline, see the [Integrating Bliss Meaning Symbols](./docs/IntegrateBlissMeaningSymbols.md) documentation.

---

### Instruction Fine-Tuning

* **[Translate btw English and Conceptual Bliss](./docs/InstructionFineTuning.md)**  
  Details the instruction fine-tuning process for translating between English and Conceptual Bliss. Includes evaluation metrics and concludes that this method is **effective**.

---

### Retrieval-Augmented Generation (RAG)

* **[Enhance Model Response Accuracy](./docs/RAG.md)**  
  Describes how Retrieval-Augmented Generation can be used to enhance model accuracy by retrieving relevant context at inference time.

---

### Context and Prompt Optimization

* **[Reflect Chat History](./docs/ReflectChatHistory.md)**  
  Discusses strategies for incorporating previous chat history into current prompts. Evaluates summarization and prompt engineering approaches, concluding that **prompt engineering is more effective**.

---

### Bliss Symbol Generation with GANs

* **[Bliss Symbol Generation Using StyleGAN2-ADA](./docs/StyleGAN2-ADATraining.md)**  
  Explains how to train a StyleGAN2-ADA model for Bliss symbol generation. Reports promising training results and viability for symbol synthesis.

* **[Bliss Symbol Generation Using StyleGAN3](./docs/StyleGAN3Training.md)**  
  Provides training results using StyleGAN3 for symbol generation. Concludes that this approach is **not effective** for Bliss symbols.

---

### Texture Inversion

* **[Symbol Manipulation](./notebooks/README.md)**  
  Describes the texture inversion technique and its potential for symbol manipulation. Evaluation shows that this method is **not effective** for the targeted use case.

See the [Texture Inversion documentation](./notebooks/README.md) for details.

## Preserving Information

### RAG (Retrieval-augmented generation)

**Conclusion**: useful

RAG (Retrieval-augmented generation) technique is explored to resolve ambiguities by retrieving relevant contextual
information from external sources, enabling the language model to generate more accurate and reliable responses.

See [RAG.md](./docs/RAG.md) for more details.

### Reflection over Chat History

**Conclusion**: useful

When users have a back-and-forth conversation, the application requires a form of "memory" to retain and
incorporate past interactions into its current processing. Two methods are explored to achieve this:

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

All utility functions are in the [`image_utils`](./image_utils) directory.

See [README.md](./image_utils/README.md) in the [`image_utils`](./image_utils) directory for details.
