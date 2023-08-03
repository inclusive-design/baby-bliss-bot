'''
Firstly install tesseract-ocr: `sudo apt-get install tesseract-ocr` on Linux and `brew install tesseract` on Mac

This script uses OCR to extract English texts from images. It loops through all images in the given directory, extract
English texts from each image. The extracted texts are saved in a txt file in the same filename and the same
directory as its corresponding image.

Before extraction, the brightness of images are enhanced by the given enhance factor to increase the extraction
accurance. If the factor is not provided, its default is 1.5. The experiment shows enhancing contrast and sharpness
doesn't help.

Usage: python extract_english_texts.py source_image_path lang enhance_factor
Parameters:
  source_image_path: The path where images are
  lang: The language code of the language to be extracted. English is "eng"
  enhance_factor: The factor value to enhance image's brightness. If not provided, the defualt value is 1.5
Return: None

Example: python extract_english_texts.py ~/Downloads/images eng 2
'''

import sys
import os
import pytesseract
from PIL import Image, ImageEnhance

# Provide the directory path as a parameter when running the script
source_image_path = sys.argv[1]
lang_code = sys.argv[2]
enhance_factor = sys.argv[3] if len(sys.argv) > 3 else 1.5

# Iterate over all images in the directory
for filename in os.listdir(source_image_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.jp2'):
        # Construct the file paths
        image_path = os.path.join(source_image_path, filename)
        text_file_path = os.path.join(source_image_path, os.path.splitext(filename)[0] + '.txt')

        # Load the image using PIL (Python Imaging Library)
        image = Image.open(image_path)

        # Enhance the image quality
        enhancer = ImageEnhance.Brightness(image)
        enhanced_image = enhancer.enhance(enhance_factor)

        # Use Tesseract OCR to extract text from the image
        extracted_text = pytesseract.image_to_string(enhanced_image, lang=lang_code)

        # Write the extracted text to a text file
        with open(text_file_path, 'w') as text_file:
            text_file.write(extracted_text)

        print(f"Extracted text from {filename} and saved it in {text_file_path}")
