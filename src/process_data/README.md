# Process Data for Blissymbolics Dictionary

This folder contains scripts to prepare and process data for adding multiple languages into the Blissymbolics dictionary JSON file. All scripts should be run from the `src/process_data` directory.

## Process Steps

### Step 1: Fix POS Colour Values

Modify `jobs/bliss-gloss/data/bliss_symbol_explanations.json` to correct the POS colour values for the following symbols (as identified by Hannes):
- 23705, 25208, 25845, 23884, 25263, 25121, 15417, 25541, 25540, 13349, 22654, 24664, 24769

### Step 2: Expand Dictionary with Multilingual Descriptions

Expand the Blissymbolics JSON dictionary with multilingual descriptions from TSV format.

```bash
python expand_bliss_dict.py ../../jobs/bliss-gloss/data/bliss_symbol_explanations.json ../data/bliss_dict/BCI-AV_SKOG_2025-02-15_multi-langs.tsv ../data/bliss_dict/bliss_symbol_explanations_multi_langs.json
```

**Inputs:**
- `bliss_symbol_explanations.json` - Original Blissymbolics symbol explanations
- `BCI-AV_SKOG_2025-02-15_multi-langs.tsv` - TSV file with multilingual descriptions

**Output:**
- `bliss_symbol_explanations_multi_langs.json` - Expanded dictionary with multiple languages

### Step 3: Clean and Standardize Glosses

Process the Blissymbolics dictionary JSON file to clean and standardize glosses.

```bash
python bliss_dict_clean_glosses.py ../data/bliss_dict/bliss_symbol_explanations_multi_langs.json ../data/bliss_dict/bliss_dict_multi_langs.json
```

**Inputs:**
- `bliss_symbol_explanations_multi_langs.json` - Expanded multilingual dictionary

**Output:**
- `bliss_dict_multi_langs.json` - Cleaned dictionary with standardized glosses

### Step 4: Identify Duplicate Glosses

Identify duplicate glosses across different items, taking into account their associated metadata such as `is_old` status and `semantics`.

```bash
python find_duplicate_glosses.py ../data/bliss_dict/bliss_dict_multi_langs.json ../data/bliss_dict/duplicate_glosses.json
```

**Inputs:**
- `bliss_dict_multi_langs.json` - Cleaned multilingual dictionary

**Output:**
- `duplicate_glosses.json` - Report of identified duplicate glosses with metadata

## Scripts

- `expand_bliss_dict.py` - Expands dictionary with multilingual data
- `bliss_dict_clean_glosses.py` - Cleans and standardizes glosses
- `find_duplicate_glosses.py` - Identifies duplicate glosses
