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

## Utility Functions

All utility functions are in `utils` directory.

### Scale down images (utils/scale_down_images.py)

This script scales down JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The output images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python scale_down_images.py [input_dir] [output_dir] [new_size]

*input_dir*: The directory where the original images are located.  
*output_dir*: The directory where the output images will be saved.  
*new_size*: The desired size of the scaled down images, in the format "widthxheight".

**Example**: python scale_down_images.py images/ scaled_down_images/ 128x128

**Returns**: None

### Parse '.wbs' file and convert to '.json' (utils/parse_wbs.py)

This script reads a winbliss file and converts each line into a json structure,
collecting all of these into an array. The json is printed to `stdout`

**Usage**: python parse_wbs.py file_path [> output_file]

*file_path*: Path to the `.wbs` file, which can be a relative path.  
*output_file*: Optional redirect path to the `.json` file.  If not given, the
JSON is printed to `stdout`.

**Example**: python parse_wbs.py ./data/wbs/some_wbs_file.wb > output.json

**Returns**: Status equal to zero if successful; equal to one if the command
failed.

## Notebooks

[`/notebooks`](./notebooks/) directory contains all notebooks that are used to train or fine-tune various models.
Each notebook usually comes with a accompanying `dockerfile.yml` to elaborate the environment that the notebook was
running in.
