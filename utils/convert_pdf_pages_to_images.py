'''
This script converts every single page in the given pdf file into individual images

Usage: python convert_pdf_pages_to_images.py source_pdf_file output_image_folder
Parameters:
  source_pdf_file: The path to the pdf file is
  output_image_folder: The directory that the converted images will be written into
Return: None

Example: python convert_pdf_pages_to_images.py ~/Downloads/icons.pdf ~/Downloads/bmw_images/
'''

import sys
import os
from pdf2image import convert_from_path

input_pdf_file = sys.argv[1]
output_dir = sys.argv[2]

pages = convert_from_path(input_pdf_file)
for count, page in enumerate(pages):
    output_file = os.path.join(output_dir, f"page_{count + 1}.jpg")
    page.save(output_file, 'JPEG')
