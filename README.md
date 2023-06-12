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

### Search and Download PDFs and JP2 zip files from the Blissymbolics archive (utils/internet_archive_scrape.py)

This script searches and downloads files from the [Blissymbolics collection](https://archive.org/details/blissymbolics)
in the Internet Archive.  The [Archive Data Extraction Feasibility](https://docs.google.com/document/d/1i9W_IXHtaaoWJD9IMA4S0d9bbYdMaCWNJRSQtFbxEhI/edit#heading=h.jszt9hbxhtuf)
document provides a five point rating scale of the archive's data in terms of how 
useful it will be for training.  The script searches and downloads archive data with ratings of 2/5 and higher based on what is noted in the feasibility document.

The script contains partial titles taken from the [Archive Data Extraction Feasibility](https://docs.google.com/document/d/1i9W_IXHtaaoWJD9IMA4S0d9bbYdMaCWNJRSQtFbxEhI/edit#heading=h.jszt9hbxhtuf) document. Each item in the Blissymbolics collection
that matches a title is used to download any PDF and any zip file of JP2 images associated with that item.

The PDF and zip files are downloaded to a subdirectory within the current directory.
Each item's identifier is used to create a subdirectory whose name is the same as the identifier.  The item's PDF and JP2 zip file are downloaded to that subdirectory.

Note that the downloader is smart enough not to download a file that was downloaded
on a previous run.  It checks if the file has matching dates and sizes and, if so,
does not download it again.  It instead reports that it is skipping that file.

**Prerequisite**: This script depends on the Internet Archive's `internetarchive`
python library.  It is recommended to first create and activate a virtual
environment and then install the library, for example:

```
$ python -m venv IA_ENV
$ source IA_ENV/bin/activate
$ pip install internetarchive
```
Complete documentation for the library is available at [The Internet
Archive Python Library](https://archive.org/developers/internetarchive/index.html) documentation site.

**Usage**: python internet_archive_scrape.py

**Returns**: None.  However, folders with names like `OTUED_8-2-1-9-1` are created,
containing `OTUED_8-2-1-9-1.pdf` and `OTUED_8-2-1-9-1_jp2.zip` files.

### Scale down images (utils/scale_down_images.py)

This script scales down JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The output images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python scale_down_images.py [input_dir] [output_dir] [new_size]

*input_dir*: The directory where the original images are located.
*output_dir*: The directory where the output images will be saved.
*new_size*: The desired size of the scaled down images, in the format "widthxheight".

**Example**: python scale_down_images.py images/ scaled_down_images/ 128x128

**Returns**: None

## Notebooks

[`/notebooks`](./notebooks/) directory contains all notebooks that are used to train or fine-tune various models.
Each notebook usually comes with a accompanying `dockerfile.yml` to elaborate the environment that the notebook was
running in.
