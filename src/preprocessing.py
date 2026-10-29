import re

from nltk.stem.isri import ISRIStemmer
from tashaphyne.stemming import ArabicLightStemmer

isri = ISRIStemmer()
light = ArabicLightStemmer()


def remove_diacritics(text):
    return re.sub(r"[ً-ْ]", "", text)


def remove_kashida(text):
    return text.replace("ـ", "")


def normalize_letters(text):
    return re.sub(r"[إأآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")


def remove_long_numbers(text):
    return re.sub(r"[0-9٠-٩]{5,}", " ", text)


def remove_digits(text):
    return re.sub(r"[0-9٠-٩]+", " ", text)


def reduce_elongation(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_punctuation(text):
    # Keeps diacritics and kashida so those steps stay independent.
    return re.sub(r"[^\w\sً-ْـ]", " ", text)


def light_stem(text):
    def stem_word(word):
        light.light_stem(word)
        return light.get_stem()
    return " ".join(stem_word(w) for w in text.split())


def stem(text):
    return " ".join(isri.stem(w) for w in text.split())


STEPS = {
    "diacritics": remove_diacritics,
    "kashida": remove_kashida,
    "letters": normalize_letters,
    "long_numbers": remove_long_numbers,
    "digits": remove_digits,
    "elongation": reduce_elongation,
    "punctuation": remove_punctuation,
    "light_stem": light_stem,
    "stem": stem,
}


def build_pipeline(names=()):
    """build_pipeline(["letters", "punctuation"]) -> function text -> text. Empty = only trims spaces."""
    names = {names} if isinstance(names, str) else set(names)

    unknown = names - STEPS.keys()
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}. Valid: {list(STEPS)}")
    if {"light_stem", "stem"} <= names:
        raise ValueError("Choose only one of 'light_stem' or 'stem'.")

    steps = [func for name, func in STEPS.items() if name in names]

    def run(text):
        for step in steps:
            text = step(text)
        return " ".join(text.split())

    return run
