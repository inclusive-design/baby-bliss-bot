# Copyright (c) 2023, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

"""
Script for searching and downloading items from the Blissymbolics archive (the
Blissymbolics collection in the Internet Archive).

This script depends on the Internet Archive's `internetarchive` python library.
It is recommended to first create and activate a virtual environment and then
install the library:
  $ python -m venv IA_ENV
  $ source IA_ENV/bin/activate
  $ pip install internetarchive

Complete documentation for the library is available at their "The Internet
Archive Python Library" documentation site:
https://archive.org/developers/internetarchive/index.html

"""
from internetarchive import search_items, download


def search_title(item_title, collection):
    """
    Search for the given title in the given collection and return the items
    found.

    The `item_title` argument is a case insenstive string that can match all or
    part of the title of an item in the collection.  For example 'newsletter'
    will match both 'Newsletter' and 'Blissword Newsletter'.

    The `collection` is a string naming the collection in which the item
    resides.  For examplele, 'blissymbolics'.

    A `Search` object is returned.
    """
    query = f'title:"{item_title}" collection:{collection}'
    print(f'Query: {query}')
    return search_items(query)


def search_and_download(item_title, collection='blissymbolics'):
    """
    Search for a given item title and, if found, download its associated items
    to the current directory.

    The `item_title` argument is a case insenstive string that can match all or
    part of the title of an item in the collection.  For example 'newsletter'
    will match 'Newsletter' and 'Blissword Newsletter'.

    The `collection` is a string naming the collection in which the item
    resides.  It defaults to 'blissymbolics'.

    Each item with the title in the collection is used to download any PDF and
    any zip file of JP2 images associated with that item.  The PDF and zip
    files are downloaded to a subdirectory within the current directory.
    Each item's identifier is used to create a subdirectory whose name is the
    same as the identifier.  Theitem's PDF and JP2 zip file are downloaded to
    that subdirectory.

    Note that the downloader is smart enough not to download a file that was
    downloaded on a previous run.  It appears to check if the files are the same
    and, if so, does not download it again.  It instead reports that it is
    skipping that file.
    """
    for item in search_title(item_title, collection):
        print('Item: ' + str(item))
        ident = item['identifier']
        download(ident, files=[f'{ident}.pdf', f'{ident}_jp2.zip'], verbose=True)


# Exact title matches -- retrieves only the items with the given title string.
# For these items, the ratings in terms of quantity of Bliss are:
# - "newspaper" 5/5
# - "Sexual" and "tell" 4/5
# - "Communicating with" 3/5
# - "Communicating together" 2/5
search_and_download('Symbol Users\' Newspaper')
search_and_download('Being Sexual: An Illustrated Series on Sexuality and Relationships')
search_and_download('After You Tell')
search_and_download('Communicating with Blissymbolics')
search_and_download('Communicating Together')

# There are many items with "newsletter" in their title and this single search
# retrieves all of them.  The ratings in parentheses are Will's ratings in terms
# of quantity of Bliss.
# - Newsletter (3.5/5)
# - BlissWorld Newsletter (3/5)
# - Symbol Coordination Committee Newsletter (2/5)
search_and_download('Newsletter')

print('Done!')
