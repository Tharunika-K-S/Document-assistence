import re
import math


# ============================================================
# WORD / SENTENCE HELPERS
# ============================================================

def get_words(text):
    """
    Extract English/Tamil words from OCR text.
    """
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0B80-\u0BFF]+",
        text or ""
    )


def get_sentences(text):
    """
    Split OCR text into sentences.
    """
    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# ENGLISH SYLLABLE ESTIMATION
# ============================================================

def count_syllables(word):
    """
    Simple English syllable estimation.
    This is intentionally lightweight so the project
    does not require another external package.
    """

    word = word.lower()

    if not word:
        return 0

    # Remove non alphabetic characters
    word = re.sub(r"[^a-z]", "", word)

    if not word:
        return 0

    vowels = "aeiouy"

    count = 0
    previous_vowel = False

    for char in word:

        is_vowel = char in vowels

        if is_vowel and not previous_vowel:
            count += 1

        previous_vowel = is_vowel

    # Silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


# ============================================================
# FLESCH READING EASE
# ============================================================

def flesch_reading_ease(text):
    """
    Calculate Flesch Reading Ease for predominantly
    English text.

    Higher score = easier to read.
    """

    words = get_words(text)
    sentences = get_sentences(text)

    english_words = [
        word for word in words
        if re.search(r"[A-Za-z]", word)
    ]

    if not english_words or not sentences:
        return None

    word_count = len(english_words)

    sentence_count = max(
        len(sentences),
        1
    )

    syllable_count = sum(
        count_syllables(word)
        for word in english_words
    )

    score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )

    return round(
        max(0, min(100, score)),
        2
    )


# ============================================================
# DIFFICULTY LEVEL
# ============================================================

def get_difficulty(score):

    if score is None:
        return "Not Available"

    if score >= 80:
        return "Very Easy"

    if score >= 60:
        return "Easy"

    if score >= 40:
        return "Moderate"

    if score >= 20:
        return "Difficult"

    return "Very Difficult"


# ============================================================
# TEXT STATISTICS
# ============================================================

def calculate_statistics(text):

    words = get_words(text)
    sentences = get_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    if sentence_count:
        average_sentence_length = (
            word_count / sentence_count
        )
    else:
        average_sentence_length = 0

    if word_count:
        average_word_length = (
            sum(len(word) for word in words)
            / word_count
        )
    else:
        average_word_length = 0

    long_words = [
        word
        for word in words
        if len(word) >= 10
    ]

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length":
            round(average_sentence_length, 2),
        "average_word_length":
            round(average_word_length, 2),
        "long_word_count":
            len(long_words)
    }


# ============================================================
# DIFFICULT WORD DETECTION
# ============================================================

COMMON_WORDS = {
    "application",
    "applicant",
    "certificate",
    "government",
    "department",
    "document",
    "information",
    "authority",
    "required",
    "submit",
    "submitted",
    "submission",
    "verification",
    "eligible",
    "eligibility",
    "procedure",
    "prescribed",
    "notification",
    "registration",
    "permission",
    "necessary",
    "supporting",
    "documents",
    "beneficiary",
    "benefit",
    "implementation",
    "administration",
    "accordingly",
    "hereby",
    "thereof",
    "whereas",
    "pursuant",
    "jurisdiction",
    "proceedings",
    "consideration",
    "disbursement",
    "documentation"
}


def find_difficult_words(text, limit=10):

    words = get_words(text)

    candidates = []

    for word in words:

        if not re.search(
            r"[A-Za-z]",
            word
        ):
            continue

        clean_word = re.sub(
            r"[^A-Za-z]",
            "",
            word
        ).lower()

        if not clean_word:
            continue

        # Long words are more likely to be difficult.
        if (
            len(clean_word) >= 9
            and clean_word not in COMMON_WORDS
        ):
            candidates.append(clean_word)

    # Remove duplicates while preserving order
    unique_words = []

    for word in candidates:

        if word not in unique_words:
            unique_words.append(word)

    return unique_words[:limit]


# ============================================================
# SIMPLE EXPLANATION
# ============================================================

SIMPLIFICATIONS = {

    "hereby": "by this document",

    "thereof": "of that",

    "therein": "in that",

    "pursuant to": "according to",

    "prior to": "before",

    "subsequent to": "after",

    "in accordance with": "following",

    "with regard to": "about",

    "in order to": "to",

    "at the earliest": "as soon as possible",

    "commence": "start",

    "commenced": "started",

    "obtain": "get",

    "obtained": "got",

    "provide": "give",

    "provided": "given",

    "require": "need",

    "required": "needed",

    "assistance": "help",

    "approximately": "about",

    "additional": "extra",

    "terminate": "end",

    "utilize": "use",

    "verify": "check",

    "verification": "checking",

    "submit": "give",

    "submission": "giving",

    "eligible": "qualified",

    "eligibility": "qualification",

    "applicant": "person applying",

    "prescribed": "specified",

    "concerning": "about",

    "regarding": "about"
}


def simplify_text(text):

    if not text:
        return ""

    simplified = text

    # Replace longer phrases first
    replacements = sorted(
        SIMPLIFICATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for original, simple in replacements:

        simplified = re.sub(
            rf"\b{re.escape(original)}\b",
            simple,
            simplified,
            flags=re.IGNORECASE
        )

    return simplified


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_readability(text):

    if not text or not text.strip():

        return {
            "score": None,
            "level": "Not Available",
            "word_count": 0,
            "sentence_count": 0,
            "average_sentence_length": 0,
            "average_word_length": 0,
            "long_word_count": 0,
            "difficult_words": [],
            "simple_explanation": ""
        }

    score = flesch_reading_ease(text)

    statistics = calculate_statistics(text)

    difficult_words = find_difficult_words(text)

    simple_explanation = simplify_text(text)

    return {

        "score": score,

        "level": get_difficulty(score),

        "word_count":
            statistics["word_count"],

        "sentence_count":
            statistics["sentence_count"],

        "average_sentence_length":
            statistics["average_sentence_length"],

        "average_word_length":
            statistics["average_word_length"],

        "long_word_count":
            statistics["long_word_count"],

        "difficult_words":
            difficult_words,

        "simple_explanation":
            simple_explanation
    }