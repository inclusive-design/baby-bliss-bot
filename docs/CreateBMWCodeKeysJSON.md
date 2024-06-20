# Create JSON for Rendering Keys on BMW Palette

This document describes two methods for creating a JSON file that is used to render all BMW code keys on the BMW
Palette.

The selection of the method is case dependent. The current [../data/bmw_palette.json](../data/bmw_palette.json)
is built using the predefined code positions method.

Note: All steps should run in the `/utils` folder.

```
cd utils
```

## Method 1: Predefined Code Positions

1. Make sure the "encoding_codes" section in bmw.json is comprehensive and correct.

```
python populate_encoding_codes.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
```

2. Compare and verify ../data/bmw-new.json from the step 1. If it's correct, use it to replace
../data/bmw.json.

3. In [../utils/create_predefined_bmw_palette_json.py](../utils/create_predefined_bmw_palette_json.py), positions
of all code keys are defined in the `code_positions` list. Every element in this list represents all code keys in
one row. `start_row` and `start_column` define which row and column the first code starts with. Adjust these values
to make sure all keys are in desired positions.

```
python create_predefined_bmw_palette_json.py ../data/bmw.json ../data/bmw_palette.json
```

## Method 2: Based on Caculated POS values

1. Make sure the "encoding_codes" section in bmw.json is comprehensive and correct.

```
python populate_encoding_codes.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
```

2. Compare and verify ../data/bmw-new.json from the step 1. If it's correct, use it to replace
../data/bmw.json.

3. Create an interim JSON file that groups encoding symbols by POS value then sort every group in
alphabetic order. 

```
python sort_by_pos.py ../data/bmw.json ../data/intermediate_BMW_conversion_data/symbols_in_pos.json
```

4. Verify these groups in ../data/intermediate_BMW_conversion_data/symbols_in_pos.json to make sure symbols are in correct groups.

5. Create the BMW code keys JSON file based on ../data/intermediate_BMW_conversion_data/symbols_in_pos.json

```
python create_bmw_palette_json.py ../data/intermediate_BMW_conversion_data/symbols_in_pos.json ../data/bmw_keys.json
```

The order of POS groups to display in a palette is defined in `pos_in_order` variable in `create_bmw_palette_json.py`.
The arrangement of code keys is to in the order of column by column.
