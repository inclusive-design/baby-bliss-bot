# Prepare Image Set For StyleGAN models

[StyleGAN model](https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/) 
offers control over the style of the generated image. Bliss has 1217 single characters that are used to compose
Bliss words. It is interesting to train StyleGAN models with these single characters to find out if they are useful to
give us some new bliss shapes for brainstorming. It will also provide the team a feel for what these systems can and
cannot do.

The image set should satisfy these conditions:
1. It contains all Bliss single characters. 
2. Images need to be transformed to grayscale. 
3. Images should be cropped or padded out to make a square of 256 x 256 with the baseline always in the same position, 
and the symbol centered horizontally.

Note: The Bliss single character with BCI ID 25600 is missing from the final image set. According to the information
from [Blissary](https://blissary.com/blissfiles/), this character is missing on purpose because it has been decided as
part of the work with encoding Blissymbolics into Unicode. This particular character will not be part of Unicode. It's
because it doesn't stand for a concept, it's just a description of a graphical shape used in the Bliss character for
squirrel.

## Steps

Step 1. At [the Bliss File page](https://blissary.com/blissfiles/), download the png package with height 344px,
transparent background, 384dpi resolution and the naming of BCI ID.

Step 2. Export [the Bliss single character](https://docs.google.com/spreadsheets/d/1t1x1UFuJC1hpjrxdXKi19Tk_Tv-9GVQWSA4sN2FScv4/edit#gid=138588066) spreadsheet as tab separated file.

Step 3. Filter out all Bliss single characters from the downloaded png package into a directory.
```
cd utils

// Filter out all Bliss single characters
python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars
Error: 25600.png not found in /Users/cindyli/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid
```

Step 4. Scan through all single characters to find the maximum dimension.
```
// Find the maximum dimensions
python get_max_dimensions.py ~/Downloads/bliss_single_chars

Results:
The max width is:  313
The list of images with the max width is:  ['26057.png', '14958.png', '24281.png', '22625.png', '17999.png', '26049.png']
The max height is:  264
The second max width is:  289
The list of images with the second max width is:  ['13090.png']
The second max height is:  0
The list of images with the second max height is:  []
```

Step 5. Resize images with the max width 313px to a width of 256px. Since all resized images need to be in the same height 
with the max dimension of 256px, it results in the calculation of the height:
```
max_height = 256 * 264 / 313 = 215.92
```

Step 6. Resize all single character images to a height of 216px.
```
// Resize
python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216

// Check the max dimension of resized images
python get_max_dimensions.py ~/Downloads/bliss_single_chars_in_height_216

Results:
The max width is:  256
The list of images with the max width is:  ['26057.png', '14958.png', '24281.png', '22625.png', '17999.png', '26049.png']
The max height is:  216
The second max width is:  236
The list of images with the second max width is:  ['13090.png']
The second max height is:  0
The list of images with the second max height is:  []

The verification shows the resizing is correct.
```

Step 7. Transform all images to grayscale. Pad out all images with the background in the same background color as the
grayscaled image to make a square of 256X256. All images are centred horizontally.
```
// Pad out all images
python image_size_sync.py ~/Downloads/bliss_single_chars_in_height_216 ~/Downloads/bliss_single_chars_final

// Verify the max dimension of final images
python get_max_dimensions.py ~/Downloads/bliss_single_chars_final
Results:
The max width is:  256
The second max width is:  0
The list of images with the second max width is:  []
The second max height is:  0
The list of images with the second max height is:  []

The verification shows the resizing is correct.
```
