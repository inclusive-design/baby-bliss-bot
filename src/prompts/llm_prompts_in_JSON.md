# LLM Prompts for Blissymbolics Sentences in JSON

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
  * If an item provides multiple English glosses (e.g., "I, me, myself"), select the one that best fits the grammatical context.
  * If an item provides a JSON object, synthesize an English word or short phrase by following this hierarchy:
    * The "classifier_info" specifies the primary concept of the word/phrase.
    * The "specifier_info" narrows down the specific meaning of the classifier.
    * The "semantics" field determines its grammatical, semantical or syntactical form.
2. Construct Sentence: Assemble the selected words into a natural, grammatically correct English sentence.
  * Apply proper verb conjugation (e.g., change "I" + "be" to "I am").
  * Fix word order if necessary for natural flow, but respect the user's intent.
3. Predict: Based on the sentence so far, generate the top 5 most likely next words to help the user complete their thought quickly.

Input List:
1. I, me, myself
2. be, am, are, is, exist
3. {
	"classifier_info": "need",
	"specifier_info": ["rest"],
	"semantics": {"POS": ["adjective", "adverb"], "category": "semantic", "features": {"modality": "null"}}
}
4. therefore, so, so that

Output Format:
Sentence so-far: {The interpreted sentence}
Top 5 Predictions:
1. {word}
2. {word}
3. {word}
4. {word}
5. {word}

**Responses**

1. Gemini 3 Pro
```
Sentence so-far: I am tired, so
Top 5 Predictions:

1. I
2. go
3. sleep
4. home
5. will
```

2. Deepseek-r1 8B
```
Sentence so-far: I need to rest.

Top 5 Predictions:
1. rest
2. good
3. tired
4. because
5. want
```

3. Llama3.1 8B
```
**Sentence So-Far:** I am going to rest.

**Top 5 Predictions:**

1. need
2. some
3. water
4. now
5. tomorrow
```

### Other Test Sentences ###

#### Example 2 ####

* **Original sentence**: The light bag was easy to carry.
* **Words in list**: light bag easy carry so
* **Input List:**

1. {
	"classifier_info": "weight",
	"specifier_info": ["littleness"],
	"semantics": {"POS": ["adjective", "adverb"], "category": "semantic", "features": {"modality": "null"}}
}
2. baggage, bag, luggage, suitcase
3. easy, easily
4. {
	"classifier_info": "thing",
	"specifier_info": ["wheel"],
	"semantics": {"POS": "verb", "category": "grammatical", "features": {"tense": "null", "voice": "null", "mood": "null", "aspect" : "null", "form": "infinitive"}}
}
5. therefore, so, so that

* **Responses**

1. Gemini 3 Pro
```
Sentence so-far: Light luggage is easy to wheel, so
Top 5 Predictions:
1. I
2. can
3. travel
4. we
5. move
```

2. Deepseek-r1 8B
```
Sentence so-far: I have a very small suitcase.

Top 5 Predictions:
1. heavy
2. light
3. large
4. small
5. bag
```

3. Llama3.1 8B
```
**Sentence So-Far:** It is lightweight.

**Top 5 Predictions:**

1. to
2. carry
3. easily
4. by
5. hand
```

#### Example 3 ####

* **Original sentence**: I finished the task and sent it for review.
* **Words in list**: I finished the task and sent it for
* **Input List:**
1. I, me, myself
2. {
  "classifier_info": "finish, complete",
  "specifier_info": [],
  "semantics": {
    "POS": "verb",
    "category": "grammatical",
    "features": {
      "tense": "past",
      "voice": "active",
      "mood": "declarative",
      "aspect": "null",
      "form": "inflected"
    }
  }
}
3. activity, male gender (in combinations)
4. and, also, plus, too
5. {
  "classifier_info": "transport, transportation",
  "specifier_info": ["departure"],
  "semantics": {
    "POS": "verb",
    "category": "grammatical",
    "features": {
      "tense": "past",
      "voice": "active",
      "mood": "declarative",
      "aspect": "null",
      "form": "inflected"
    }
  }
}
6. for (the purpose of), in order to

* **Responses**

1. Gemini 3 Pro
```
Sentence so-far: I finished work and left to
Top 5 Predictions:
1. go
2. meet
3. catch
4. see
5. buy
```

2. Deepseek-r1 8B
```
**Sentence so-far:**  
I finished an activity and departed.

**Top 5 Predictions:**  
1. home  
2. school  
3. work  
4. trip  
5. station
```

3. Llama3.1 8B
```
**Sentence So-Far:** I finished.

**Top 5 Predictions:**

1. an
2. activity
3. to travel
4. by
5. car
```
