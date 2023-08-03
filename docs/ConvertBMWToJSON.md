# Convert BMW Encoding to JSON Structure

This document describes steps for converting digitalized BMW encoding documents to a JSON file
that will serve as the foundation for implementing the BMW input method.

BMW encoding documents are in PDF format. These PDFs are composed by digitalized images of orginal
books. The coversio method is:

1. Split every single page in a PDF into .jpg files
2. Use OCR library to extract texts from .jpg files
3. Verify text files to correct missing or wrong texts
4. Convert BMW encoding in text files into one single JSON file

## Steps

All steps should be run in the `/utils` folder.

```
cd utils
```

1. Split every page of the PDF into an individual .jpg image.

```
python convert_pdf_pages_to_images.py ~/Downloads/icons.pdf ~/Downloads/bmw_images/
```

2. Extract English texts from .jpg files and write into .txt files.

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
