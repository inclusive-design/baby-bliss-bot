# Create JSON for Rendering Keys on BMW Palette

This document describes steps for creating a JSON file that is used to render all keys on the BMW
Palette.

Note: All steps should run in the `/utils` folder.

```
cd utils
```

1. Make sure the "encoding_symbols" section in bmw.json is comprehensive and correct.

```
python populate_encoding_symbols.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
```

2. Compare and verify ../data/bmw-new.json from the step 1. If it's correct, use it to replace
../data/bmw.json.

3. Create an interim JSON file that groups encoding symbols by POS value then sort every group in
alphabetic order. 

```
python sort_by_pos.py ../data/bmw.json ../data/intermediate_BMW_conversion_data/symbols_in_pos.json
```

4. Verify these groups in ../data/intermediate_BMW_conversion_data/symbols_in_pos.json to make sure symbols are in correct groups.

5. Create the BMW keys JSON file based on ../data/intermediate_BMW_conversion_data/symbols_in_pos.json

```
python create_keys_json.py ../data/intermediate_BMW_conversion_data/symbols_in_pos.json ../data/bmw_keys.json
```

The order of POS groups to display in a palette is defined in `pos_in_order` variable in `create_keys_json.py`.
The arrangement of keys is to in the order of column by column.
