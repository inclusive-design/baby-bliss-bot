'''
This script reads bmw.json, find all messages that have null BCI-AV-ID values, use Spacy to parse and transform
these messages to conceptual Bliss, then find their BCI-AV-IDs. This script handles messages in these formats:
1. Verb in different form.
For example: "begin", "to begin", "beginning", "began", "begun", "begins"  all share the same Bliss symbol of
its infinitive form "begin".

2. Plural nouns.
For example: "books" -> [book, ";", 9011].

3. Subject + Pronoun.
For example: "I am" -> [I, be]
"I were" -> [past_tense, I, be]
"I will" -> [future_tense, I]
"he isn't" -> [he, not, be]
"isn't he" -> [question_mark, he, not, be]
"should he" -> [question_mark, past_tense, he]
"shouldn't he" -> [question_mark, past_tense, he, not]
When the BCI-AV-ID for a word in the tranformed sentence cannot be found, an error will be reported.

Note: The code for each case above should be uncommented and ran one by one. The result from each run should be
checked carefully to ensure its correctness.

Usage: python fill_in_null_bliss_id_with_spacy.py source_bmw_path bliss_explanation_json_location output_bmw_path
Parameters:
  source_bmw_path: The path where bmw.json is
  bliss_explanation_json_location: The location of the JSON file that contains the translation between Bliss
  BCI-AV-ID and its language translation
  output_bmw_path: The path of the output BMW file
Return: None

Example: python fill_in_null_bliss_id_with_spacy.py ../data/bmw.json ../data/bliss_symbol_explanations.json ../data/bmw-new.json
'''

import json
import sys
import spacy


def find_infinitive_form_for_verb(text):
    doc = nlp(text)
    infinitive_form = None

    for token in doc:
        # handle cases such as "swim", "swimming", "swims", "swam", "begun", "began" etc
        if token.pos_ == "VERB":
            infinitive_form = token.lemma_
            break  # Stop at the first verb encountered
        # handle the case of "to swim"
        elif token.dep_ == "inf" and token.head.lemma_ == "to" and token.pos_ == "VERB":
            infinitive_form = token.text
            break

    return infinitive_form


def find_bliss_id_for_verb(text, bliss_explanation_json):
    for item in bliss_explanation_json:
        description = item["description"].lower()
        if (text + "-" in description or text + ";" in description) and "(to)" in description:
            print(text + "; " + item["id"] + "; " + description)
            return int(item["id"])

    return None


def find_infinitive_form_for_plural_noun(text):
    doc = nlp(text)
    infinitive_form = None

    for token in doc:
        if token.pos_ == "NOUN" and token.tag_ == "NNS":
            infinitive_form = token.lemma_

    return infinitive_form


def find_bliss_id_in_general(text, bliss_explanation_json):
    map = {
        "do": 13860,
        "can": 13114,
        "be": 12639,
        "not": 15733,
        "they": 17713,
        "we": 18212,
        "i": 14916,
        "he": 14687,
        "she": 16494,
        "believe": 12661,
        "may": 16226,
        "past_tense": 27105,
        "future_tense": 27057,
        "question_mark": 8485
    }

    for key, value in map.items():
        if text.lower() == key.lower():
            return value

    for item in bliss_explanation_json:
        defs = item["description"].lower().split(",")

        # Skip old descriptions
        if defs[-1].endswith("_(OLD)"):
            continue
        # The suffix "-(to)" indicates a verb. It's not part of the definition
        if defs[-1].endswith("-(to)"):
            defs[-1] = defs[-1][:-6]

        if text in defs:
            # print(text + "; " + item["id"] + "; " + item["description"].lower())
            return int(item["id"])

    return None


def get_sequence_for_msg_with_subject(text):
    doc = nlp(text)
    sequence = []
    is_past_tense = False
    is_future_tense = False
    is_question = False
    has_subject = False
    is_subject = False
    has_not = False
    position = 0

    for token in doc:
        if token.dep_ == "subj" or token.dep_ == "nsubj":
            has_subject = True
            is_subject = True
        if token.lemma_ == "not":
            has_not = True
        if token.tag_ == "VBD" or token.tag_ == "VBN" or token.lemma_ == "have" or token.lemma_ == "had":
            is_past_tense = True
        if token.lemma_ in ["will", "shall"]:
            is_future_tense = True
        if token.lemma_ in ["do", "be", "have", "had", "can", "could", "would", "may", "might", "should"] and position == 0:
            is_question = True
        if token.lemma_ == "could" or token.lemma_ == "should" or token.lemma_ == "would" or token.lemma_ == "might":
            is_past_tense = True
            position = position + 1
            if token.lemma_ == "could":
                sequence.append("can")
            elif token.lemma_ == "should" or token.lemma_ == "would" or token.lemma_ == "might":
                sequence.append("will")
            continue

        # Move the subject in a question to the first element in the array
        if is_subject and position > 0:
            sequence.insert(0, token.lemma_)
        else:
            sequence.append(token.lemma_)

        position = position + 1
        is_subject = False

    if has_not:
        # switch "not" value with the element in front of it
        not_index = sequence.index("not")
        sequence[not_index], sequence[not_index - 1] = sequence[not_index - 1], sequence[not_index]
    if is_past_tense:
        sequence.insert(0, "past_tense")
    if is_future_tense:
        sequence.insert(0, "future_tense")
    if is_question:
        sequence.insert(0, "question_mark")

    # this must be done after switching "not" value with the element in front it
    # to ensure the intact sequence structure at switching "not"
    words_to_remove = ["have", "had", "will", "shall"]
    sequence = [word for word in sequence if word not in words_to_remove]

    return sequence if len(sequence) > 1 and has_subject else None


source_json_file = sys.argv[1]
bliss_explanation_json_location = sys.argv[2]
output_json_location = sys.argv[3]

with open(source_json_file, 'r') as file:
    data = json.load(file)
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # load bliss translation json file
    with open(bliss_explanation_json_location, 'r') as file:
        bliss_explanation_json = json.load(file)

    for message, value in data["encodings"].items():
        if value["bci-av-id"] is None:
            # 1. handle single words
            if message.startswith("to ") or len(message.split()) == 1:
                # 1.1. Handle verb in various forms
                # For example, "begin", "to begin", "beginning", "began", "begun", "begins"
                # should all use the Bliss symbol for "begin"
                infinitive_form_for_verb = find_infinitive_form_for_verb(message)
                if infinitive_form_for_verb is not None:
                    value["bci-av-id"] = find_bliss_id_for_verb(infinitive_form_for_verb, bliss_explanation_json)

                # 1.2. Handle noun in plural form
                # For example, the BCI-AV-ID for "books" should be [{id_for_book}, ";", 9011]
                infinitive_form_for_noun = find_infinitive_form_for_plural_noun(message)
                if infinitive_form_for_noun is not None:
                    bliss_id_for_noun = find_bliss_id_in_general(infinitive_form_for_noun, bliss_explanation_json)
                    if bliss_id_for_noun is not None:
                        value["bci-av-id"] = [bliss_id_for_noun, ";", 9011]

            # 2. Handle multiple words messages with a subject such as "I should", "I shouldn't", "should I", "shouldn't I"
            bliss_sequence = get_sequence_for_msg_with_subject(message)
            if bliss_sequence is not None:
                encoding = []
                position = 0
                for text in bliss_sequence:
                    bliss_id = find_bliss_id_in_general(text, bliss_explanation_json)
                    if bliss_id is not None:
                        encoding.append(bliss_id)
                        if (position < len(bliss_sequence) - 1):
                            encoding.append("//")
                    else:
                        print("Error: ", message, ": cannot find bliss id for \"", text, "\"")
                    position = position + 1
                value["bci-av-id"] = encoding

# Write the JSON into a file
with open(output_json_location, "w") as json_file:
    json_file.write(json.dumps(data, indent=4))
print(f"The final JSON is written into {output_json_location}")
