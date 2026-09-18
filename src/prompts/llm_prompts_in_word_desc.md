# LLM Prompts for Blissymbolics Sentences in Word Description

This document explores effective LLM prompts for interpreting Blissymbolics (an AAC system), converting symbol sequences into grammatically correct English sentences, and predicting the user's next words.

## Test Sentences

1. I’m tired, so I’m going to rest for a while.
2. The light bag was easy to carry.
3. I finished the task and sent it for review.


## Prediction Prompt ##

**Example 1**
* **Original sentence**: I am tired so
* **Words in list**: I am tired so
* **Prompt**

System Role:
You are an AI assistant for a user communicating via Blissymbolics (an AAC system). Your task is to interpret a stream of raw symbol data, convert it into a grammatically correct English sentence, and predict what the user will say next.

Task Logic:
1. Analyze Input: Read the list of items below. The list represents the sequence of symbols pressed by the user.
  * If an item provides multiple English glosses (e.g., "I, me, myself"), select the one that fits the grammatical context.
  * If an item provides a semantic description (e.g., category: "need", description: "rest", POS: "adjective"), synthesize these clues into the single specific English word that matches that definition (e.g., "tired").
2. Construct Sentence: Assemble the selected words into a natural, grammatically correct English sentence.
  * Apply proper verb conjugation (e.g., change "I" + "be" to "I am").
  * Fix word order if necessary for natural flow, but respect the user's intent.
3. Predict: Based on the sentence so far, generate the top 5 most likely next words to help the user complete their thought quickly.

Input List:
1. I, me, myself
2. be, am, are, is, exist
3. This word or a short phase is classified as "need", being described by "rest", the POS attribute is "adjective" or "adverb"
4. therefore, so, so that

Output Format:
Sentence so-far: {The interpreted sentence}
Top 5 Predictions:
{word}
{word}
{word}
{word}
{word}

**Responses**

1. Gemini 3 Pro
```
Sentence so-far: I am tired so
Top 5 Predictions:
I
goodnight
will
going
can
```

2. Deepseek-r1 8B
```
Sentence so-far: I am tired therefore...

Top 5 Predictions:
why
want
ask
need
help
```

3. Llama3.1 8B
```
Here's my interpretation:

**Analysis**

1. I, me, myself
	* Based on the grammatical context (implying a subject), I select: **I**
2. be, am, are, is, exist
	* The user selects: **am** (a verb conjugation to match "I" as the subject)
3. This word or a short phase is classified as "need", being described by "rest", POS "adjective" or "adverb"
	* Based on the semantic description, I synthesize this into a specific English word that matches the definition: 
**tired**
4. therefore, so, so that
	* The user selects: **so** (a conjunction)

**Construct Sentence**

Assembling the selected words, I construct a sentence:

"I am tired."

This is a simple and grammatically correct sentence.

**Predict Next Words**

Based on the sentence so far, I predict the top 5 most likely next words:

1. **because**
2. **so**
3. **that**
4. **I**
5. **need**

The user may continue expressing their sentiment about being tired or provide additional context for why they are 
feeling that way.
```

### Other Test Sentences ###

#### Example 2 ####

* **Original sentence**: The light bag was easy to carry.
* **Words in list**: light bag easy carry so
* **Input List:**

1. This word or a short phase is classified as "weight", being described by "littleness", the POS attribute is "adjective" or "adverb"
2. baggage, bag, luggage, suitcase
3. easy, easily
4. This word or a short phase is classified as "thing", being described by "wheel", the POS attribute is "infinitive form of verb"
5. therefore, so, so that

* **Responses**

1. Gemini 3 Pro
```
Sentence so-far: Light luggage is easy to wheel, so
Top 5 Predictions:
I
we
it
can
let's
```

2. Deepseek-r1 8B
```
Sentence so-far: Light heavy easy to wheel therefore...

Top 5 Predictions:
wheel
barrow
move
lift
push
```

3. Llama3.1 8B
```
Here's my interpretation:

**Analysis**

1. "weight" described as "littleness", POS "adjective" or "adverb"
	* Based on the semantic description, I synthesize this into a specific English word that matches the definition: 
**light**
2. baggage, bag, luggage, suitcase
	* The user has selected one of these options; based on context and grammatical flow, I infer they have chosen: 
**baggage** (a common noun)
3. easy, easily
	* The user selects: **easily** (an adverb)
4. "thing" described by "wheel", POS "infinitive form of verb"
	* Based on the semantic description, I synthesize this into a specific English word that matches the definition: 
**to wheel**
5. therefore, so, so that
	* The user selects: **so** (a conjunction)

**Construct Sentence**

Assembling the selected words, I construct a sentence:

"I have light baggage and can easily move it because some things need to be wheeled."

However, this is not grammatically correct; we should rephrase to respect subject-verb agreement. Let's revise:

"My light bag is easy to carry and so are my other belongings."

**Predict Next Words**

Based on the sentence so far, I predict the top 5 most likely next words:

1. **that**
2. **because**
3. **they**
4. **can**
5. **should**
```

#### Example 3 ####

* **Original sentence**: I finished the task and sent it for review.
* **Words in list**: I finished the task and sent it for
* **Input List:**
1. I, me, myself
2. This word or a short phase is classified as "finish, complete", the POS attribute is "verb", the tense attribute is "past"
3. activity, male gender (in combinations)
4. and, also, plus, too
5. This word or a short phase is classified as "transport, transportation", being described by "departure", the POS attribute is "verb", the tense attribute is "past"
6. for (the purpose of), in order to

* **Responses**

1. Gemini 3 Pro
```
Sentence so-far: I finished work and left for
Top 5 Predictions:
home
the
school
vacation
dinner
```

2. Deepseek-r1 8B
```
Sentence so-far: I finished male activity and departed for...

Top 5 Predictions:
work
reason
home
continue
return
```

3. Llama3.1 8B
```
Here's my interpretation:

**Analysis**

1. I, me, myself
	* Based on the grammatical context (implying a subject), I select: **I**
2. This word or a short phase is classified as "finish, complete", POS "verb", tense attribute "past"
	* Based on the semantic description and part-of-speech attributes, I synthesize this into a specific English word 
that matches the definition: **finished**
3. activity, male gender (in combinations)
	* The user selects: **activity** (a noun), but given the context of the sentence, it's more likely they intended 
to select **he** or **him**, implying a masculine subject (e.g., "He was doing an activity")
4. and, also, plus, too
	* The user selects: **and** (a conjunction)
5. This word or a short phase is classified as "transport, transportation", being described by "departure", POS 
"verb", tense attribute "past"
	* Based on the semantic description and part-of-speech attributes, I synthesize this into a specific English word 
that matches the definition: **departed**
6. for (the purpose of), in order to
	* The user selects: **for** (a preposition indicating purpose)

**Construct Sentence**

Assembling the selected words, I construct a sentence:

"I finished an activity and he departed for... "

However, this is still incomplete; we need to add more context to finish the sentence.

**Predict Next Words**

Based on the sentence so far, I predict the top 5 most likely next words:

1. **order**
2. **appointment**
3. **meeting**
4. **event**
5. **business**
```
