# Convert BMW Encoding to JSON Structure

This document describes steps for converting digitized BMW encoding documents to a JSON file
that will serve as the foundation for implementing the BMW input method.

BMW encoding documents are in PDF format. These PDFs are composed by digitized images of orginal
books. The coversion method is:

1. Split each page in the PDF into its own .jpg file
2. Use OCR library to extract texts from .jpg files. The OCR library extracts text from images
into .txt files. See the documentation of [Extract English Texts from Images](/utils/README.md#extract-english-texts-from-images-utilsextract_english_textspy)
for details
3. Verify text files to correct missing or wrong texts
4. Convert BMW encoding in text files into one single JSON file

## Steps

All steps should be run in the `/utils` folder.

```
cd utils
```

1. Split each page in the PDF into its own .jpg file

```
python convert_pdf_pages_to_images.py ~/Downloads/icons.pdf ~/Downloads/bmw_images/
```

2. Extract English texts from .jpg files and write into .txt files

```
python extract_english_texts.py ~/Downloads/bmw_images/ eng
```

Modify .txt file to remove headings and footers that are not BMW encoding. Correct wrong texts if found any.

3. Convert encoding in all .txt files into one JSON file

```
python convert_bmw_to_json.py ~/Downloads/bmw_texts/ ../data/bliss_symbol_explanations.json ../data/bmw.json ../data/error.txt
```

The converted JSON file is at `../data/bmw.json`. Errors are written into `../data/error.json`. Based on the
information in the error file, correct errors and re-run the script until there isn't any error.

See [`utils/README.md`](../utils/README.md) for details about scripts used above.
