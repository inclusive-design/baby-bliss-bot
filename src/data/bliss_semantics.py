"""
===============================================================================
README — Blissymbolics Linguistic Annotation Schema
===============================================================================
This file defines metadata and semantics for Blissymbolics indicators
and modifiers. It uses a structured annotation schema to represent
linguistic information such as part of speech, grammatical features,
semantic shifts, and usage notes.

-------------------------------------------------------------------------------
Core Attributes
-------------------------------------------------------------------------------

Each entry may contain the following attributes:

1. Type: Optional. Specifies the kind of annotation the symbol represents.
Valid values (exactly one):
     - POS          : Part of speech (e.g., noun, verb)
     - TYPE_SHIFT   : Transforms one POS into another (e.g., verb → noun)

2. Type Value: Optional. Identifies the specific value for the selected Type.
   - POS values: "noun", "verb", "adjective", "adverb" (Exactly one)
   - TYPE_SHIFT value: "concretization"

3. Category: Optional. Broad linguistic grouping. Valid values (one or more):
     - "grammatical"
     - "semantic"
     - "syntactical"

4. Features: Fine‑grained linguistic properties. Available features depend on
POS and may have single or multiple values, as specified below.

-------------------------------------------------------------------------------
Features by Part of Speech
-------------------------------------------------------------------------------
VERBS
-----
- tense: Locates an action in time.
Valid values: "null" | "past" | "present" | "future" (one)

- voice: Shows relationship between the subject and action.
Valid values: "null" | "active" | "passive" (one)

- mood: Expresses attitude or intent.
Valid values: "declarative" | "conditional" | "imperative" (one)
Note: mood may vary language to language on how its used. Declarative is
assumed unless question/exclamation markers are present.

- aspect: Indicates how an action occurs over time.
Valid values: "continuous"

- form: Variations of verbs and nouns.
Valid values: "finite" | "infinitive" | "present-participle" | "past-participle-1" |
 "past-participle-2" | "gerund" (one)
Note: If tense and mood are "null", the verb is treated as non-finite (infinitive or participle).

- negation: "without" | "not" | "opposite" (one)

NOUNS
-----
- number: "singular" | "plural" (one)

- definiteness: Identifies a specific or general thing.
Valid values: "indefinite" | "definite" (one)
Example: "an apple" (indefinite), "the apple" (definite)

- gender: "neutral" | "feminine" | "masculine" (one)

- person: "first-person" | "second-person" | "third-person" (one)

- size: "diminutive"

- possessive: "possessor"

- position: "pre" | "post" (one or more)
Syntax note:
    - pre: modifier before head (e.g., "colour of the car")
    - post : modifier after head (e.g., "car's colour")

- default-position  : "pre" | "post" (one)

- quantifier: "many, much" | "all" | "any" | "both" | "each, every" | "either" | "neither" | "half" | "quarter" | "one third" | "two thirds" | "three quarters" | "several" (one)

- link: "association" | "derivative" (one)
Example:
    - furniture ↔ chair (association)
    - province → country (derivative)

- time: "ago, then (past)" | "now" | "then_future, so, later" (one)
Note: Attaches to nouns but functions adverbially.

- numeric: "zero" → "nine" (one)

- negation: "without" | "not" | "opposite" (one)

- relational: "same, equal, equality" | "blissymbol part" | "part of" | "about, concerning, regarding, in relation to" | "across" | "after, behind" | "against, opposed to" | "along with" | "among" | "around" | "at" | "before, in front of, prior to" | "between" | "by, by means of, of" | "on" | "out of (forward)" | "out of (downward)" | "out of (upward)" | "out of (backward)" | "into (forward)" | "into (downward)" | "into (upward)" | "into (backward)" | "outside" | "inside" | "over, above" | "under, below" | "under (ground level)" | "instead" | "for the purpose of, in order to" | "from" | "to, toward" | "through" | "until" | "belongs to" (one)

- concept-transforming: "similar to" | "look similar to" | "sound similar to" | "same sound" | "generalization" (one)

ADJECTIVES & ADVERBS
-------------------
- modality: Represents whether something is possible or realized.
Valid values: "potential" | "completed" (one)

- degree: "intensity" | "more (comparative)" | "most (comparative)" | "comparative less" | "minimum" (one)

- negation: "without" | "not" | "opposite" (one)

OTHER
-----
- structural-marker: "combine marker" | "what" (one)
- possible-role: "modifier" | "specifier" (one or more)

-------------------------------------------------------------------------------
Additional Metadata
-------------------------------------------------------------------------------

- equivalent_modifier / equivalent_indicator: References the ID of an equivalent
Blissymbolics indicator or modifier. (one)

- priority: Determines processing precedence. Represented as a list of IDs ordered
from highest to lowest priority.

- position: default position when it is not between other characters.
Valid values: "prefix" | "suffix" (one)
Example: "peace: opposite + war" (prefix), "danger: creation + intensity" (suffix)

Note: Action and description indicators are commonly used across users, while
present‑action and adverb indicators are more typical in full‑form usage.
"""

# Blissymbolics Indicators
INDICATOR_SEMANTICS = {
    # action indicators
    # infinitive verb or present tense verb; similar to ID: 928 (includes tense as present), here is doesn't include tense
    # BCI-AV ID: 8993
    "81": {
        "POS": "verb",
        "category": "grammatical",
        "features": {
            "form": "infinitive"
        },
        "priority": ["81", "928"]
    },
    # active verb
    # BCI-AV ID: 8994
    "82": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English present conditional form
    # BCI-AV ID: 8995
    "83": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "conditional", "form": "finite"}
    },

    # description indicators
    # the equivalent of the English -ed or -en ending
    # BCI-AV ID: 8996
    "84": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "completed"}
    },
    # equivalent to English words ending in -able
    # BCI-AV ID: 8997
    "85": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "potential"}
    },
    # the equivalent of English adjectives/adverbs
    # BCI-AV ID: 8998
    "86": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "priority": ["86", "902"]
    },
    # back to action indicators
    # the equivalent of the English future tense
    # BCI-AV ID: 8999
    "87": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English future conditional form
    # BCI-AV ID: 9000
    "88": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English future passive form
    # BCI-AV ID: 9001
    "89": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English future passive conditional form
    # BCI-AV ID: 9002
    "90": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "conditional", "form": "finite"}
    },
    # something is being acted upon
    # BCI-AV ID: 9003
    "91": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English past tense
    # BCI-AV ID: 9004
    "92": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English past conditional form
    # BCI-AV ID: 9005
    "93": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English past passive conditional form
    # BCI-AV ID: 9006
    "94": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English past passive form
    # BCI-AV ID: 9007
    "95": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English present passive conditional form
    # BCI-AV ID: 9008
    "96": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "conditional", "form": "finite"}
    },

    # represent a concrete object
    # BCI-AV ID: 9009
    "97": {
        "and": [{
            "POS": "noun",
            "category": "grammatical"
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },

    # represent multiple concrete objects
    # BCI-AV ID: 9010
    "98": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"number": "plural"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    # BCI-AV ID: 9011
    "99": {
        "category": "grammatical",
        "features": {"number": "plural"}
    },
    # BCI-AV ID: 24667
    "904": {
        "category": "grammatical",
        "features": {"definiteness": "definite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # the female modifier (ID: 314) is used more. Indicator is not used in communication
    # BCI-AV ID: 24668
    "905": {
        "category": "grammatical",
        "features": {"gender": "feminine", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "314",
        "priority": ["314", "905"]
    },
    # BCI-AV ID: 12335
    "106": {
        "category": "grammatical",
        "features": {"gender": "masculine", "number": "singular"}
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    # BCI-AV ID: 24669
    "906": {
        "category": "grammatical",
        "features": {"person": "first-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "10",
        "priority": ["10", "906"]
    },
    # the past participle form
    # BCI-AV ID: 28044
    "5996": {
        "category": "grammatical",
        "features": {"number": "plural", "definiteness": "definite"}
    },
    # BCI-AV ID: 28045
    "5997": {
        "and": [{
            "category": "grammatical",
            "features": {"definiteness": "definite", "number": "singular"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    # BCI-AV ID: 28046
    "5998": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"number": "plural", "definiteness": "definite"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },

    # indicator (adverb)
    # BCI-AV ID: 24665
    "902": {
        "POS": "adverb",
        "category": "grammatical",
        "notes": "for teaching purposes",
        "priority": ["86", "902"]
    },
    # similar to ID: 81;
    # BCI-AV ID: 24807
    "928": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "mood": "declarative", "form": "finite"},
        "notes": "for teaching purposes",
        "priority": ["81", "928"]
    },
    # the diminutive modifier is used more. Indicator (ID: 5999) is not used
    # BCI-AV ID: 25458
    "992": {
        "category": "grammatical",
        "features": {"size": "diminutive", "form": "finite"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "5999",
        "priority": ["5999", "992"]
    },
    # imperative mood
    # BCI-AV ID: 24670
    "907": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"mood": "imperative", "form": "finite"}
    },
    # 3 participles
    # BCI-AV ID: 24674
    "911": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-1"},
        "notes": "for teaching purposes"
    },
    # BCI-AV ID: 24675
    "912": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-2"},
        "notes": "for teaching purposes"
    },
    # BCI-AV ID: 24677
    "914": {
        "POS": ["verb", "adjective"],
        "category": "grammatical",
        "features": {"form": "present-participle"},
        "notes": "for teaching purposes"
    },
    # back to nouns
    # BCI-AV ID: 24671
    "908": {
        "category": "grammatical",
        "features": {"definiteness": "indefinite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # BCI-AV ID: 24672
    "909": {
        "category": "grammatical",
        "features": {"gender": "neutral", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    # BCI-AV ID: 24678
    "915": {
        "category": "grammatical",
        "features": {"person": "second-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "11",
        "priority": ["11", "915"]
    },
    # BCI-AV ID: 24679
    "916": {
        "category": "grammatical",
        "features": {"person": "third-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "12",
        "priority": ["12", "916"]
    },
    # continuous indicator
    # BCI-AV ID: 28043
    "903": {
         "POS": "noun",
         "category": "grammatical",
         "features": {
             "form": "gerund"
         },
         "notes": "Primary indicator for noun-ING (gerunds).",
         # first priority is the continuous indicator, then second priority is the present tense indicator
         "priority": ["903", "82"]
     },
    # possessive indicator; both indicator and modifier (ID: 160) are used, but modifier is used more in English (opposite is true for Swedish).
    # BCI-AV ID: 24676
    "913": {
        "POS": "noun",
        "category": ["grammatical", "syntactical"],
        "features": {
            "grammatical": {"possessive": "possessor"},
            "syntactical": {
               "position": ["pre", "post"],
               "default-position": "post"
            },
        },
        "notes": "for teaching purposes",
        "equivalent_modifier": "160",
        "priority": ["160", "913"]
    },
    # object form; can use object form with or without indicator - is an alternative, modifier (ID: 6003) has never been used
    # BCI-AV ID: 24673
    "910": {
        "POS": "noun",
        "category": "syntactical",
        "features": {"position": ["pre", "post"], "default-position": "post"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "6003",
        "priority": ["optional", "910", "6003"]
    },
}

# Blissymbolics Modifiers
MODIFIER_SEMANTICS = {
    # BCI-AV ID: 14166
    "314": {
        "features": {
           "gender": "feminine",
           "number": "singular",
           "position": "suffix"
        },
        "equivalent_indicator": "905",
        "priority": ["314", "905"]
    },
    # BCI-AV ID: 8497
    "10": {
        "or": [{
            "features": {
                "person": "first-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "906",
            "priority": ["10", "906"],
        }, {
            "numeric": "one",
            "features": {
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
        }]
    },
    # BCI-AV ID: 8498
    "11": {
        "or": [{
            "features": {
                "person": "second-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "915",
            "priority": ["11", "915"],
        }, {
            "features": {
                "numeric": "two",
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
        }]
    },
    # BCI-AV ID: 8499
    "12": {
         "or": [{
            "features": {
                "person": "third-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "916",
            "priority": ["12", "916"],
          }, {
            "features": {
                "numeric": "three",
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
         }]
    },
    # BCI-AV ID: 28052
    "5999": {
        "features": {
            "size": "diminutive",
            "position": "suffix"
        },
        "equivalent_indicator": "992",
        "priority": ["5999", "992"],
    },

    # BCI-AV ID: 12352
    "112": {
        "time": "ago, then (past)",
        "features": {
           "position": "suffix"
        }
    },

    # BCI-AV ID: 17705
    "648": {
        "time": "then_future, so, later",
        "features": {
           "position": "suffix"
        }
    },

    # BCI-AV ID: 15736
    "474": {
        "time": "now",
        "features": {
           "position": "suffix"
        }
    },

    # Structural markers
    # BCI-AV ID: 13382
    "233": {
        "structural-marker": "combine marker",
        "notes": "special case (combine marker acts like quotation marks surrounding a set of symbols)",
        "position": ["prefix", "suffix"]
    },

    # What
    # BCI-AV ID: 18229
    "699": {
        "structural-marker": "what",
        "features": {
           "position": "prefix"
        },
        "notes": "interrogative when used as a prefix, otherwise a specifier"
    },

    # Scalar degree operators
    # BCI-AV ID: 14947
    "401": {
        "degree": "intensity",
        "features": {
           "position": "prefix"
        },
        "notes": "exclamatory when used as a prefix, otherwise a specifier"
    },
    # BCI-AV ID: 24879
    "937": {
        "degree": "more (comparative)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 24944
    "968": {
        "degree": "most (comparative)",
        "features": {
           "position": "prefix"
        }
    },

    # BCI-AV ID: none (not yet in the official BCI-AV)
    "6438": {
        "degree": "comparative less",
        "features": {
           "position": "prefix"
        }
    },

    # BCI-AV ID: 29405
    "6321": {
        "degree": "minimum",
        "features": {
           "position": "prefix"
        }
    },

    # Identity-affecting operators
    # "B449/B401"
    # BCI-AV ID: 15733
    "2088": {
        "negation": "not, negative, no, don't, doesn't",
        "features": {
           "position": "prefix"
        },
        "priority": ["449", "2088", "486"]
    },
    # BCI-AV ID: 15927
    "486": {
        "negation": "opposite",
        "features": {
           "position": "prefix"
        },
        "priority": ["449", "2088", "486"]
    },
    # Concept-transforming operators
    # "B1060/B578"
    # BCI-AV ID: 16984
    "2404": {
        "concept-transforming": "similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B1060/B578/B303"
    # BCI-AV ID: 16985
    "2405": {
        "concept-transforming": "look similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B1060/B578/B608"
    # BCI-AV ID: 16986
    "2406": {
        "concept-transforming": "sound similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B578/B608"
    # BCI-AV ID: 16714
    "2312": {
        "concept-transforming": "same sound",
        "features": {
           "position": "prefix"
        }
    },
    # "B578/B303": "look same" but missing in the BCI-AV
    # BCI-AV ID: 14430
    "348": {
        "concept-transforming": "generalization",
        "features": {
           "link": "association",
           "position": "prefix"
        }
    },
    # Relational operators
    # BCI-AV ID: 15474
    "449": {
        "negation": "minus, no, without",
        "features": {
           "position": "prefix"
        },
        "priority": ["449", "2088", "486"]
    },
    # BCI-AV ID: 16713
    "578": {
        "relational": "same, equal, equality",
        "features": {
           "position": "prefix"
        }
    },
    # "B502/B167"
    # BCI-AV ID: 12858
    "1309": {
        "relational": "blissymbol part",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15972
    "502": {
        "relational": "part of",
        "features": {
           "link": "derivative",
           "position": "prefix"
        },
        "notes": "position is prefix (modifier) when describing part of/component of X (e.g. tonsils are a part of the throat, gene is part of DNA). Position is suffix (specifier) when describing X into parts, divided into/produces components (e.g. suit, jigsaw puzzle)"
    },
    # BCI-AV ID: 12324
    "102": {
        "relational": "about, concerning, regarding, in relation to",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12333
    "104": {
        "relational": "across",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12348
    "109": {
        "relational": "after, behind",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12351
    "111": {
        "relational": "against, opposed to",
        "features": {
           "position": "prefix"
        },
        "notes": "Position is prefix (most cases), suffix (when specifying what type)"
    },
    # "B120/B120"
    # BCI-AV ID: 12364
    "1189": {
        "relational": "along with",
        "features": {
           "position": "prefix",
        }
    },
    # "B162/B368"
    # BCI-AV ID: 25653
    "5274": {
        "relational": "among",
        "features": {
           "position": "prefix"
        },
        "notes": "Related meanings: between, to, inside"
    },
    # BCI-AV ID: 12580
    "134": {
        "relational": "around",
        "features": {
           "position": "prefix"
        },
    },
    # BCI-AV ID: 12591
    "135": {
        "relational": "at",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12656
    "158": {
        "relational": "before, in front of, prior to",
        "features": {
           "position": "prefix"
        },
    },
    # BCI-AV ID: 12669
    "162": {
        "relational": "between",
        "features": {
           "position": "prefix"
        },
    },
    # BCI-AV ID: 13100
    "195": {
        "relational": "by, by means of, of",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15918
    "482": {
        "relational": "on",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15943
    "491": {
        "relational": "out of (forward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15944
    "492": {
        "relational": "out of (downward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25134
    "977": {
        "relational": "out of (upward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25133
    "976": {
        "relational": "out of (backward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 14952
    "402": {
        "relational": "into (forward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25895
    "1124": {
        "relational": "into (downward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25896
    "1125": {
        "relational": "into (upward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25894
    "1123": {
        "relational": "into (backward)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15942
    "490": {
        "relational": "outside",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 14932
    "398": {
        "relational": "inside",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 15948
    "493": {
        "relational": "over, above",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 17969
    "676": {
        "relational": "under, below",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 25628
    "1102": {
        "relational": "under (ground level)",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 14381
    "331": {
        "relational": "instead",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 14382
    "332": {
        "relational": "for the purpose of, in order to",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 14403
    "337": {
        "relational": "from",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 17739
    "657": {
        "relational": "to, toward",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 17724
    "653": {
        "relational": "through",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 17982
    "677": {
        "relational": "until",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12663
    "160": {
        "relational": "belongs to",
        "features": {
           "position": "prefix"
        },
        "equivalent_indicator": "913",
        "priority":  ["160", "913"],
        "notes": "Position role is primarily suffix as modifier, but can also be prefix as specifier."
    },
    # Quantifiers
    # "B368"
    # prefix modifier
    # BCI-AV ID: 14647
    "368": {
        "quantifier": "many, much",
        "features": {
           "position": "prefix"
        }
    },
    # pending: few (not yet in bliss-glyph-data.js)
    # BCI-AV ID: 12360
    "117": {
        "quantifier": "all",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 12321
    "100": {
        "quantifier": "any",
        "features": {
           "position": "prefix"
        }
    },
    # "B11/B117"
    # BCI-AV ID: 12879
    "1324": {
        "quantifier": "both",
        "features": {
           "position": "prefix"
        }
    },
    # "B10/B117"
    # BCI-AV ID: 13893
    "1580": {
        "quantifier": "each, every",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 13914
    "286": {
        "quantifier": "either",
        "features": {
           "position": "prefix"
        }
    },
    # "B449/B286"
    # BCI-AV ID: 15706
    "2066": {
        "quantifier": "neither",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 24906
    "951": {
        "quantifier": "half",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 24932
    "962": {
        "quantifier": "quarter",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 26064
    "1151": {
        "quantifier": "one third",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 26065
    "1152": {
        "quantifier": "two thirds",
        "features": {
           "position": "prefix"
        }
    },
    # BCI-AV ID: 26066
    "1153": {
        "quantifier": "three quarters",
        "features": {
           "position": "prefix"
        }
    },
    # "B559/B11"
    # BCI-AV ID: 16762
    "2353": {
        "quantifier": "several",
        "features": {
           "position": "prefix"
        },
        "notes": "position is prefix (inferred by related meaning: many/much)"
    },
    # BCI-AV ID: 8496
    "9": {
        "numeric": "zero",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8500
    "13": {
        "numeric": "four",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8501
    "14": {
        "numeric": "five",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8502
    "15": {
        "numeric": "six",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8503
    "16": {
        "numeric": "seven",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8504
    "17": {
        "numeric": "eight",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # BCI-AV ID: 8505
    "18": {
        "numeric": "nine",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    }
}
