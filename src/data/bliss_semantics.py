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
    # infinitive verb or present tense verb; similar to ID: 24807 (includes tense as present), here is doesn't include tense
    "8993": {
        "POS": "verb",
        "category": "grammatical",
        "features": {
            "form": "infinitive"
        },
        "priority": ["8993", "24807"]
    },
    # active verb
    "8994": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English present conditional form
    "8995": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "conditional", "form": "finite"}
    },

    # description indicators
    # the equivalent of the English -ed or -en ending
    "8996": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "completed"}
    },
    # equivalent to English words ending in -able
    "8997": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "potential"}
    },
    # the equivalent of English adjectives/adverbs
    "8998": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "priority": ["8998", "24665"]
    },
    # back to action indicators
    # the equivalent of the English future tense
    "8999": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English future conditional form
    "9000": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English future passive form
    "9001": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English future passive conditional form
    "9002": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "conditional", "form": "finite"}
    },
    # something is being acted upon
    "9003": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English past tense
    "9004": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English past conditional form
    "9005": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English past passive conditional form
    "9006": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "conditional", "form": "finite"}
    },
    # the equivalent of the English past passive form
    "9007": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "declarative", "form": "finite"}
    },
    # the equivalent of the English present passive conditional form
    "9008": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "conditional", "form": "finite"}
    },

    # represent a concrete object
    "9009": {
        "and": [{
            "POS": "noun",
            "category": "grammatical"
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },

    # represent multiple concrete objects
    "9010": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"number": "plural"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    "9011": {
        "category": "grammatical",
        "features": {"number": "plural"}
    },
    "24667": {
        "category": "grammatical",
        "features": {"definiteness": "definite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # the female modifier (ID: 14166) is used more. Indicator is not used in communication
    "24668": {
        "category": "grammatical",
        "features": {"gender": "feminine", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "14166",
        "priority": ["14166", "24668"]
    },
    "12335": {
        "category": "grammatical",
        "features": {"gender": "masculine", "number": "singular"}
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24669": {
        "category": "grammatical",
        "features": {"person": "first-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8497",
        "priority": ["8497", "24669"]
    },
    # the past participle form
    "28044": {
        "category": "grammatical",
        "features": {"number": "plural", "definiteness": "definite"}
    },
    "28045": {
        "and": [{
            "category": "grammatical",
            "features": {"definiteness": "definite", "number": "singular"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    "28046": {
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
    "24665": {
        "POS": "adverb",
        "category": "grammatical",
        "notes": "for teaching purposes",
        "priority": ["8998", "24665"]
    },
    # similar to ID: 8993;
    "24807": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "mood": "declarative", "form": "finite"},
        "notes": "for teaching purposes",
        "priority": ["8993", "24807"]
    },
    # the diminutive modifier is used more. Indicator (ID: 28052) is not used
    "25458": {
        "category": "grammatical",
        "features": {"size": "diminutive", "form": "finite"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28052",
        "priority": ["28052", "25458"]
    },
    # imperative mood
    "24670": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"mood": "imperative", "form": "finite"}
    },
    # 3 participles
    "24674": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-1"},
        "notes": "for teaching purposes"
    },
    "24675": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-2"},
        "notes": "for teaching purposes"
    },
    "24677": {
        "POS": ["verb", "adjective"],
        "category": "grammatical",
        "features": {"form": "present-participle"},
        "notes": "for teaching purposes"
    },
    # back to nouns
    "24671": {
        "category": "grammatical",
        "features": {"definiteness": "indefinite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    "24672": {
        "category": "grammatical",
        "features": {"gender": "neutral", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24678": {
        "category": "grammatical",
        "features": {"person": "second-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8498",
        "priority": ["8498", "24678"]
    },
    "24679": {
        "category": "grammatical",
        "features": {"person": "third-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8499",
        "priority": ["8499", "24679"]
    },
    # continuous indicator
    "28043": {
         "POS": "noun",
         "category": "grammatical",
         "features": {
             "form": "gerund"
         },
         "notes": "Primary indicator for noun-ING (gerunds).",
         # first priority is the continuous indicator, then second priority is the present tense indicator
         "priority": ["28043", "8994"] 
     },
    # possessive indicator; both indicator and modifier (ID: 12663) are used, but modifier is used more in English (opposite is true for Swedish).
    "24676": {
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
        "equivalent_modifier": "12663",
        "priority": ["12663", "24676"]
    },
    # object form; can use object form with or without indicator - is an alternative, modifier (ID: 28057) has never been used
    "24673": {
        "POS": "noun",
        "category": "syntactical",
        "features": {"position": ["pre", "post"], "default-position": "post"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28057",
        "priority": ["optional", "24673", "28057"]
    },
}

# Blissymbolics Modifiers
MODIFIER_SEMANTICS = {
    # "B314"
    "14166": {
        "features": {
           "gender": "feminine",
           "number": "singular",
           "position": "suffix"
        },
        "equivalent_indicator": "24668",
        "priority": ["14166", "24668"]
    },
    # "B10"
    "8497": {
        "or": [{
            "features": {
                "person": "first-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "24669",
            "priority": ["8497", "24669"],
        }, {
            "numeric": "one",
            "features": {
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
        }]
    },
    # "B11"
    "8498": {
        "or": [{
            "features": {
                "person": "second-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "24678",
            "priority": ["8498", "24678"],
        }, {
            "features": {
                "numeric": "two",
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
        }]
    },
    # "B12"
    "8499": {
         "or": [{
            "features": {
                "person": "third-person",
                "number": "singular",
                "position": "suffix"
            },
            "equivalent_indicator": "24679",
            "priority": ["8499", "24679"],
          }, {
            "features": {
                "numeric": "three",
                "position": "prefix"
            },
            "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
         }]
    },
    # "B5999"
    "28052": {
        "features": {
            "size": "diminutive",
            "position": "suffix"
        },
        "equivalent_indicator": "25458",
        "priority": ["28052", "25458"],
    },

    # "B112"
    "12352": {
        "time": "ago, then (past)",
        "features": {
           "position": "suffix"
        }
    },

    # "B648"
    "17705": {
        "time": "then_future, so, later",
        "features": {
           "position": "suffix"
        }
    },

    # "B474"
    "15736": {
        "time": "now",
        "features": {
           "position": "suffix"
        }
    },

    # Structural markers
    # "B233"
    "13382": {
        "structural-marker": "combine marker",
        "notes": "special case (combine marker acts like quotation marks surrounding a set of symbols)",
        "position": ["prefix", "suffix"]
    },

    # What
    # "B699"
    "18229": {
        "structural-marker": "what",
        "features": {
           "position": "prefix"
        },
        "notes": "interrogative when used as a prefix, otherwise a specifier"
    },

    # Scalar degree operators
    # "B401"
    "14947": {
        "degree": "intensity",
        "features": {
           "position": "prefix"
        },
        "notes": "exclamatory when used as a prefix, otherwise a specifier"
    },
    # "B937"
    "24879": {
        "degree": "more (comparative)",
        "features": {
           "position": "prefix"
        }
    },
    # "B968"
    "24944": {
        "degree": "most (comparative)",
        "features": {
           "position": "prefix"
        }
    },

     # "B6438"
    "24944": {
        "degree": "comparative less",
        "features": {
           "position": "prefix"
        }
    },

     # "B6321"
    "24944": {
        "degree": "minimum",
        "features": {
           "position": "prefix"
        }
    },
     
    # Identity-affecting operators
    # "B449/B401"
    "15733": {
        "negation": "not, negative, no, don't, doesn't",
        "features": {
           "position": "prefix"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # "B486"
    "15927": {
        "negation": "opposite",
        "features": {
           "position": "prefix"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # Concept-transforming operators
    # "B1060/B578"
    "16984": {
        "concept-transforming": "similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B1060/B578/B303"
    "16985": {
        "concept-transforming": "look similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B1060/B578/B608"
    "16986": {
        "concept-transforming": "sound similar to",
        "features": {
           "position": "prefix"
        }
    },
    # "B578/B608"
    "16714": {
        "concept-transforming": "same sound",
        "features": {
           "position": "prefix"
        }
    },
    # "B578/B303": "look same" but missing in the BCI-AV
    # "B348"
    "14430": {
        "concept-transforming": "generalization",
        "features": {
           "link": "association",
           "position": "prefix"
        }
    },
    # Relational operators
    # "B449"
    "15474": {
        "negation": "minus, no, without",
        "features": {
           "position": "prefix"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # "B578"
    "16713": {
        "relational": "same, equal, equality",
        "features": {
           "position": "prefix"
        }
    },
    # "B502/B167"
    "12858": {
        "relational": "blissymbol part",
        "features": {
           "position": "prefix"
        }
    },
    # "B502"
    "15972": {
        "relational": "part of",
        "features": {
           "link": "derivative",
           "position": "prefix"
        },
        "notes": "position is prefix (modifier) when describing part of/component of X (e.g. tonsils are a part of the throat, gene is part of DNA). Position is suffix (specifier) when describing X into parts, divided into/produces components (e.g. suit, jigsaw puzzle)"
    },
    # "B102"
    "12324": {
        "relational": "about, concerning, regarding, in relation to",
        "features": {
           "position": "prefix"
        }
    },
    # "B104"
    "12333": {
        "relational": "across",
        "features": {
           "position": "prefix"
        }
    },
    # "B109"
    "12348": {
        "relational": "after, behind",
        "features": {
           "position": "prefix"
        }
    },
    # "B111"
    "12351": {
        "relational": "against, opposed to",
        "features": {
           "position": "prefix"
        },
        "notes": "Position is prefix (most cases), suffix (when specifying what type)"
    },
    # "B120/B120"
    "12364": {
        "relational": "along with",
        "features": {
           "position": "prefix",
        }
    },
    # "B162/B368"
    "25653": {
        "relational": "among",
        "features": {
           "position": "prefix"
        },
        "notes": "Related meanings: between, to, inside"
    },
    # "B134"
    "12580": {
        "relational": "around",
        "features": {
           "position": "prefix"
        },
    },
    # "B135"
    "12591": {
        "relational": "at",
        "features": {
           "position": "prefix"
        }
    },
    # "B158"
    "12656": {
        "relational": "before, in front of, prior to",
        "features": {
           "position": "prefix"
        },
    },
    # "B162"
    "12669": {
        "relational": "between",
        "features": {
           "position": "prefix"
        },
    },
    # "B195"
    "13100": {
        "relational": "by, by means of, of",
        "features": {
           "position": "prefix"
        }
    },
    # "B482"
    "15918": {
        "relational": "on",
        "features": {
           "position": "prefix"
        }
    },
    # "B491"
    "15943": {
        "relational": "out of (forward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B492"
    "15944": {
        "relational": "out of (downward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B977"
    "25134": {
        "relational": "out of (upward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B976"
    "25133": {
        "relational": "out of (backward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B402"
    "14952": {
        "relational": "into (forward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B1124"
    "25895": {
        "relational": "into (downward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B1125"
    "25896": {
        "relational": "into (upward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B1123"
    "25894": {
        "relational": "into (backward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B490"
    "15942": {
        "relational": "outside",
        "features": {
           "position": "prefix"
        }
    },
    # "B398"
    "14932": {
        "relational": "inside",
        "features": {
           "position": "prefix"
        }
    },
    # "B493"
    "15948": {
        "relational": "over, above",
        "features": {
           "position": "prefix"
        }
    },
    # "B676"
    "17969": {
        "relational": "under, below",
        "features": {
           "position": "prefix"
        }
    },
    # "B1102"
    "25628": {
        "relational": "under (ground level)",
        "features": {
           "position": "prefix"
        }
    },
    # "B331"
    "14381": {
        "relational": "instead",
        "features": {
           "position": "prefix"
        }
    },
    # "B332"
    "14382": {
        "relational": "for the purpose of, in order to",
        "features": {
           "position": "prefix"
        }
    },
    # "B337"
    "14403": {
        "relational": "from",
        "features": {
           "position": "prefix"
        }
    },
    # "B657"
    "17739": {
        "relational": "to, toward",
        "features": {
           "position": "prefix"
        }
    },
    # "B653"
    "17724": {
        "relational": "through",
        "features": {
           "position": "prefix"
        }
    },
    # "B677"
    "17982": {
        "relational": "until",
        "features": {
           "position": "prefix"
        }
    },
    # "B160"
    "12663": {
        "relational": "belongs to",
        "features": {
           "position": "prefix"
        },
        "equivalent_indicator": "24676",
        "priority":  ["12663", "24676"],
        "notes": "Position role is primarily suffix as modifier, but can also be prefix as specifier."
    },
    # Quantifiers
    # "B368"
    # prefix modifier
    "14647": {
        "quantifier": "many, much",
        "features": {
           "position": "prefix"
        }
    },
    # pending: few (not yet in bliss-glyph-data.js)
    # "B117"
    "12360": {
        "quantifier": "all",
        "features": {
           "position": "prefix"
        }
    },
    # "B100"
    "12321": {
        "quantifier": "any",
        "features": {
           "position": "prefix"
        }
    },
    # "B11/B117"
    "12879": {
        "quantifier": "both",
        "features": {
           "position": "prefix"
        }
    },
    # "B10/B117"
    "13893": {
        "quantifier": "each, every",
        "features": {
           "position": "prefix"
        }
    },
    # "B286"
    "13914": {
        "quantifier": "either",
        "features": {
           "position": "prefix"
        }
    },
    # "B449/B286"
    "15706": {
        "quantifier": "neither",
        "features": {
           "position": "prefix"
        }
    },
    # "B951"
    "24906": {
        "quantifier": "half",
        "features": {
           "position": "prefix"
        }
    },
    # "B962"
    "24932": {
        "quantifier": "quarter",
        "features": {
           "position": "prefix"
        }
    },
    # "B1151"
    "26064": {
        "quantifier": "one third",
        "features": {
           "position": "prefix"
        }
    },
    # "B1152"
    "26065": {
        "quantifier": "two thirds",
        "features": {
           "position": "prefix"
        }
    },
    # "B1153"
    "26066": {
        "quantifier": "three quarters",
        "features": {
           "position": "prefix"
        }
    },
    # "B559/B11"
    "16762": {
        "quantifier": "several",
        "features": {
           "position": "prefix"
        },
        "notes": "position is prefix (inferred by related meaning: many/much)"
    },
    # "B9"
    "8496": {
        "numeric": "zero",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B13"
    "8500": {
        "numeric": "four",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B14"
    "8501": {
        "numeric": "five",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B15"
    "8502": {
        "numeric": "six",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B16"
    "8503": {
        "numeric": "seven",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B17"
    "8504": {
        "numeric": "eight",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B18"
    "8505": {
        "numeric": "nine",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    }
}
