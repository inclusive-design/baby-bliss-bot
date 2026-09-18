Prompts used to generate various datasets:

1. Positive context sentences:

# Complete positive context sentences
Generate a Python list named "positive_context_sentences" containing 100 high-quality English sentences. Each sentence must include one of these words: "out of", "exit". A sentence may contain multiple occurrences of the word, but the last occurrence must appear naturally toward the end of the sentence.

Each sentence should be:
* Contextually rich, describing a realistic or meaningful scenario.
* Grammatically diverse, spanning a wide range of structures, such as:
* Conditional sentences (e.g., if-clauses)
* Relative clauses
* Passive voice constructions
* Direct and indirect questions
* Compound and complex sentences
* Imperative and exclamatory forms

The dataset should reflect a broad range of real-world contexts, including but not limited to:
* Personal, professional, and academic settings
* Emotions, relationships, and interpersonal interactions
* Scientific, philosophical, and technical topics
* Everyday activities and abstract reflections

The output must be a valid Python list of unique strings, each representing one sentence, and assigned to the variable "positive_context_sentences". Format your response as Python code only, with no additional explanation or comments.

Generate a Python list named "negative_context_sentences" containing 100 sentences. Each sentence must include either the word "afraid", "frightened" or "scared", used in various contexts except in the context of height. Ensure each sentence is long and contextually rich, naturally leading up to the keyword, which should appear towards the end of the sentence. The set of sentences should include a wide variety of grammatical structures (e.g., conditionals, relative clauses, passive voice, questions, compound/complex sentences etc) and a diverse range of real-world scenarios (e.g., architecture, nature, human height, machinery, etc.). Output the result as a valid Python list of strings.

# Partial positive context sentences
Generate a Python list named "positive_context_sentences" containing 100 context-rich, incomplete sentences. Each partial sentence must lead to the strong continuation with one of these words: "dispersion", "dissemination", "scattering", "spread", "spreading". Such continuations should be grammatically or semantically correct.

Each sentence should be:
* Contextually rich, describing a realistic or meaningful scenario.
* Grammatically diverse, spanning a wide range of structures, such as:
* Conditional sentences (e.g., if-clauses)
* Relative clauses
* Passive voice constructions
* Direct and indirect questions
* Compound and complex sentences
* Imperative and exclamatory forms

The dataset should reflect a broad range of real-world contexts, including but not limited to:
* Personal, professional, and academic settings
* Emotions, relationships, and interpersonal interactions
* Scientific, philosophical, and technical topics
* Everyday activities and abstract reflections

Do not include ellipses, placeholders, or unnatural truncation. Return the result as valid Python code: a list named "positive_context_sentences" containing string elements.

2. Negative context sentences:
# Glosses used in other contexts
Generate a Python list named "negative_context_sentences" containing 20 sentences. Each sentence must include the word "level", used in various contexts except in the context of building levels. Ensure each sentence is long and contextually rich, naturally leading up to the keyword, which should appear towards the end of the sentence. The set of sentences should include a wide variety of grammatical structures (e.g., conditionals, relative clauses, passive voice, questions, compound/complex sentences etc) and a diverse range of real-world scenarios (e.g., architecture, nature, human height, machinery, etc.). Output the result as a valid Python list of strings.

# Glosses are unlikely to be used
Generate a Python list named "negative_context_sentences" containing 100 context-rich, incomplete sentences. Each sentence must end naturally but remain clearly unfinished, inviting continuation. Critically, the continuation must not begin with any of these words: "dispersion", "dissemination", "scattering", "spread", "spreading". Such continuations should be grammatically or semantically implausible.

Each sentence should be:
* Contextually rich, describing a realistic or meaningful scenario.
* Grammatically diverse, spanning a wide range of structures, such as:
* Conditional sentences (e.g., if-clauses)
* Relative clauses
* Passive voice constructions
* Direct and indirect questions
* Compound and complex sentences
* Imperative and exclamatory forms

The dataset should reflect a broad range of real-world contexts, including but not limited to:
* Personal, professional, and academic settings
* Emotions, relationships, and interpersonal interactions
* Scientific, philosophical, and technical topics
* Everyday activities and abstract reflections

Do not include ellipses, placeholders, or unnatural truncation. Return the result as valid Python code: a list named "negative_context_sentences" containing string elements.

3. Fine tuning sentences:
Generate a dataset for fine-tuning. I need 100 English texts that each contain one or more occurrences of these words as a noun: "break", "fracture", "injury" and "damage". These sentences must reflect a wide variety of contexts and structures, including:

* Different tenses (past, present, future, perfect, etc.)
* Moods (indicative, imperative, subjunctive, conditional)
* Sentence types (declarative, interrogative, exclamatory, imperative)
* Sentence lengths (very short to long, complex sentences)
* Contexts (daily life, academic, professional, casual, emotional, hypothetical, etc.)
* Grammatical diversity (subject, object, determiner use, negations, subordinate clauses, etc.)
* Various text lengths

The output should be a valid Python list of unique strings assigned to the variable "fine_tuning_sentences". Each string must represent a text.

Output format: Python code only. Do not include any comments or explanations.

4. Text generation prompts for testing:
I am evaluating a fine-tuned language model’s ability to understand and generate text based on articles and determiners the use of these words: "dispersion", "dissemination", "scattering", "spread", "spreading". Generate 10 diverse partial sentence prompts containing a "{placeholder}" near the beginning of each sentence, where the placeholder represents one of these target words. Each prompt should be a unique, incomplete sentence designed to elicit natural continuations. Ensure stylistic and topical diversity across prompts.

Output the result as valid Python code, assigning the list of strings to the variable "testing_text_generation_prompts". Do not include any comments, explanations, or additional text—output the code only.
