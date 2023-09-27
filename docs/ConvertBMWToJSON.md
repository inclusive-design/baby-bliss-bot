# Convert BMW Encoding to JSON Structure

This document describes steps for converting digitized BMW encoding documents to a JSON file
that will serve as the foundation for implementing the BMW input method.

BMW encoding documents are in PDF format. These PDFs are composed by digitized images of orginal
books. The coversion method is:

Note: All steps should run in the `/utils` folder.

```
cd utils
```

1. Split each page in the 
[03 - ICON SORT.pdf](https://drive.google.com/file/d/1pxaTjymmVRcD0kJ15sZyuc9HEVHCK31F/view?usp=drive_link)
into individual .jpg file

```
python convert_pdf_pages_to_images.py ~/Downloads/icons.pdf ~/Downloads/bmw_images/
```

2. Use OCR library to extract texts from .jpg files. The OCR library extracts text from images
into .txt files. See the documentation of [Extract English Texts from Images](/utils/README.md#extract-english-texts-from-images-utilsextract_english_textspy)
for details. Verify text files to correct missing or wrong texts. These corrected text files are
saved in the directory 
[data/intermediate_BMW_conversion_data/bmw_texts](../data/intermediate_BMW_conversion_data/bmw_texts).
Note: as the script extract the meaning part by looking for the first letter with lower case.
Some meaning parts in the pdf start with upper case such as "TV", "I'm hungry". During the correction
process, the first letter or all letters of this kind are changed to lower case in order to satisfying
the script requirement. Since these changes should be reverted in the final JSON file, they are
tracked in the file
[data/intermediate_BMW_conversion_data/special_handling.txt](../data/intermediate_BMW_conversion_data/special_handling.txt).

```
python extract_english_texts.py ~/Downloads/bmw_images/ eng
```

3. Convert BMW encoding in text files into one single JSON file

```
python convert_bmw_to_json.py ~/Downloads/bmw_texts/ ../data/bliss_symbol_explanations.json ../data/bmw.json ../data/error.txt
```

4. Mannually modify the JSON file to revert changes tracked in 
[data/intermediate_BMW_conversion_data/special_handling.txt](../data/intermediate_BMW_conversion_data/special_handling.txt).
5. Run a script to loop through the JSON file, find messages with null BCI-AV-IDs, parse and transform
every message to accommodating Bliss, then compose BCI-AV-ID based on the transformed message.

```
python fill_in_null_bliss_id_with_spacy.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
```

Run utility script `find_null_ids.py` to find messages that still have null BCI-AV-IDs:

6. Manually fill in the rest messages that have null BCI-AV-IDs.

## Utility Scripts ##

### Find messages that BCI-AV-ID is null (./find_null_ids.py)

```
python find_null_ids.py ../data/bmw.json
```

### Populate "encoding_symbols" section in bmw.json

```
python populate_encoding_symbols.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
```

### Find missing encodings from BMW text file directory

```
python find_missing_encodings.py ../data/bmw.json ../data/intermediate_BMW_conversion_data/bmw_texts/ ../data/bliss_symbol_explanations.json missing_encodings.json error.txt
```

See [`utils/README.md`](../utils/README.md) for details about scripts used above.
