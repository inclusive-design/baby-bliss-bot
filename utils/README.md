# Utility Scripts

This directory contains all utility scripts that are used for preparing dataset, training models etc.

## Get Bliss single characters (get_bliss_single_chars.py)

This script filters out all Bliss single characters from a directory with all Bliss symbols.

**Usage**: python script_name.py [tsv_file_path] [all_bliss_symbol_dir] [target_dir]

* *tsv_file_path*: The path to the .tsv file to be read. This file contains single characters by BCI IDs
* *all_bliss_symbol_dir*: The path to the directory where all Bliss symbol images are located.
* *target_dir*: The path to the directory where matched symbol images will be copied to.

**Example**: python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars

**Return**: None

## Resize all images to a same height (resize_images_to_same_height.py)

This script resizes all images in a directory to the same height. The resized images are saved into a target directory.

**Usage**: python resize_images_to_same_height.py [image_dir] [target_height] [target_dir]

* *image_dir*: The directory with all images
* *target_height*: The target height to resize all images to
* *target_dir*: The target directory to save resized images

**Example**: python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216

**Return**: None

## Get max image dimensions (get_max_dimensions.py)

This script finds the maximum width and maximum height of all PNG and JPG images in a directory,
along with a list of image filenames that have the maximum width and maximum height.
It also returns the second maximum width and second maximum height, along with their respective
lists of image filenames.

**Usage**: python get_max_dimensions.py [image_directory]

* *image_directory*: The path to the directory containing the images.

**Example**: python get_max_dimensions.py images/

**Return**: tuple: A tuple containing:
* the maximum width (int)
* maximum height (int)
* a list of filenames of images with maximum width (list)
* a list of filenames of images with maximum height (list)
* the second maximum width (int), the second maximum height (int)
* a list of filenames of images with the second maximum width (list)
* a list of filenames of images with the second maximum height (list)

## Scale down images (scale_down_images.py)

This script scales down JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The output images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python scale_down_images.py [input_dir] [output_dir] [new_size]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.
* *new_size*: The desired size of the scaled down images, in the format "widthxheight".

**Example**: python scale_down_images.py images/ scaled_down_images/ 128x128

**Return**: None

## Sync up image sizes (image_size_sync.py)

This script synchronizes the size of all PNG and JPG files in the input directory.
It first finds the maximum dimension (either width or height) among all the input images.
Then it loops through the image directory to perform these operations for every image:
1. Transform the image to grayscale and find the background color of this image using the color code at the pixel
(1, 1);
2. Create a square canvas with the maximum dimension as its width and height. The color of the canvas is the background
color observed at the previous step;
3. Copy each input image onto the center of the canvas, without changing the size of the input image. This ensures that
each output image has the same maximum dimension and is centered in the canvas. 
Finally, all output images are saved in the specified output directory.

**Usage**: python image_size_sync.py [input_dir] [output_dir]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.

**Example**: python image_size_sync.py images/ output/

**Return**: None

## Convert a PDF File to Individual Images (utils/convert_pdf_pages_to_images.py)

This script converts every single page in the given pdf file into individual images

**Usage**: python convert_pdf_pages_to_images.py source_pdf_file output_image_folder
Parameters:
  source_pdf_file: The path to the pdf file is
  output_image_folder: The directory that the converted images will be written into
Return: None

**Example**: python convert_pdf_pages_to_images.py ~/Downloads/icons.pdf ~/Downloads/bmw_images/

**Returns**: None

## Extract English Texts from Images (utils/extract_english_texts.py)

This script uses OCR to extract English texts from images. It loops through all images in the given directory, extract
English texts from each image. The extracted texts are saved in a txt file in the same filename and the same
directory as its corresponding image.

Before extraction, the brightness of images are enhanced by the given enhance factor to increase the extraction
accurance. If the factor is not provided, its default is 1.5. The experiment shows enhancing contrast and sharpness
doesn't help.

**Prerequisite**: Firstly install [`tesseract-ocr`](https://github.com/tesseract-ocr/tesseract)
* On Unix, run: `sudo apt-get install tesseract-ocr`
* On Mac, run: `brew install tesseract` 

**Usage**: python extract_english_texts.py [source_image_dir] [lang_code] [enhance_factor]

*source_image_path*: The path where images are
*lang*: The language code of the language to be extracted. English is "eng"
*enhance_factor*: The factor value to enhance image's brightness. If not provided, the defualt value is 1.5

**Example**: python extract_english_texts.py ~/Downloads/images eng 2

**Returns**: None

## Convert BMW encoding list into JSON (utils/convert_bmw_to_json.py)

This script reads txt files that contain BMW encoding list from a directory and convert them into JSON.
The script will report error when BCI-AV-ID for a text is not found or the target message is not found.
At the end of execution, a set of texts that don't have the matching BCI-AV-ID will be summarized and
reported.

Errors will be written into the given error file. The converted JSON will be written into the given JSON file.

These errors are reported:
1. The length of icons in an encoding is less than 2 or longer than 4.
2. The message conveyed by the encoding is not found. This is because the message should be in lower case
   in the scanned document. When all texts are in upper case, this error is reported.

**Usage**: python convert_bmw_to_json.py source_txt_path bliss_explanation_json_location output_json_location output_error_location

*source_txt_path*: The path where text files are
*bliss_explanation_json_location*: The location of the JSON file that contains the translation between Bliss
BCI-AV-ID and its language translation
*output_json_location*: The location of the output JSON file. If it doesn't exist, the script will create it
*output_error_location*: The location of the error file that rows in input files not processed due to any error are written into

**Example**: python convert_bmw_to_json.py ~/Downloads/bmw-text ../data/bliss_symbol_explanations.json bmw.json error.txt

**Return**: None

**File formats**

1. Sample content of a .txt file in the `source_txt_path` directory
```
SAY THINK VERB+S talks
SAY TO+VERB to say
SAY VERB say
```
Using the first line `SAY THINK VERB+S talks` as an example, "SAY", "THINK" and "VERB+S" are three keys pressed on the
keyboard. It delivers a message "talks".

2. The generated JSON structure for the BMW encoding is:
```
[
    "encodings": {
        "talked": {
            "encoding": [
                "SAY",
                "THINK",
                "VERB+ED"
            ],
            "bci-av-id": 123
        }
    ...
    },
     "word_to_id_map": {
        "VERB": 12335,
        "VERB+S": [12335, "/", 8499],
        ...
    }
]
```

**Note** 

Regarding values for BCI-AV-ID information, such as `encodings.talked.bci-av-id` and `word_to_id_map.VERB`
can be an array if this Bliss symbol is composed by multiple symbols. The use of "/" or ";" means:
* In the format of [12335, "/", 8499], the Bliss character 12335 and the Bliss character of 8499 are
displayed side by side;
* In the format of [12335, ";", 8499], the Bliss character 12335 and the indicator 8499 are displayed
in the way that the indicator 8499 is on top of the the charactor 12355;


## Fill in null BCI-AV-ID values for messages using SpaCy (utils/fill_in_null_bliss_id_with_spacy.py)

This script reads bmw.json, find all messages that have null BCI-AV-ID values, use Spacy to parse and transform
these messages to accommodating Bliss, then find their BCI-AV-IDs. This script handles messages in these formats:
1. Verb in different form.
For example: "begin", "to begin", "beginning", "began", "begun", "begins"  all share the same Bliss symbol of
its infinitive form "begin".

2. Plural nouns. 
For example: "books" -> [book, ";", 9011].

3. Subject + Pronoun. The script supports two transformations:
3.1. Transform to Conceptual Bliss
For example: "I am" -> [I, be]
"I were" -> [past_tense, I, be]
"I will" -> [future_tense, I]
"he isn't" -> [he, not, be]
"isn't he" -> [question_mark, he, not, be]
"should he" -> [question_mark, past_tense, he]
"shouldn't he" -> [question_mark, past_tense, he, not]

3.2. Transform to Accommadating Bliss in English
For example: "I am" -> [I, am]
"I were" -> [I, were]
"he isn't" -> [he, is, not]
"isn't he" -> [is, not, he]

When the BCI-AV-ID for a word in the tranformed sentence cannot be found, an error will be reported.

Note: The code for each case above should be uncommented and ran one by one. The result from each run should be
checked carefully to ensure its correctness.

**Usage**: python fill_in_null_bliss_id_with_spacy.py source_bmw_path bliss_explanation_json_location output_bmw_path

*source_bmw_path*: The path where bmw.json is
*bliss_explanation_json_location*: The location of the JSON file that contains the translation between Bliss
BCI-AV-ID and its language translation
*output_bmw_path*: The path of the output BMW file

**Example**: python fill_in_null_bliss_id_with_spacy.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json

**Return**: None

## Find messages with null BCI-AV-ID values (utils/find_null_ids.py)

Loops through bmw.json and reports all messages whose BCI-AV-ID is null.

**Usage**: python find_null_ids.py source_bmw_path
*source_bmw_path*: The path where bmw.json is

**Example**: python find_null_ids.py ../data/bmw.json

**Return**: None
