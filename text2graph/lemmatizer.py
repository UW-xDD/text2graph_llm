from typing import Protocol
from enum import Enum

import nltk
import spacy
from nltk.stem import WordNetLemmatizer

from .llm import ask_llm


class Implementation(Enum):
    NLTK = "nltk"
    SPACY = "spacy"
    LLM = "llm"


class Lemmatizer(Protocol):
    def __init__(self, implementation: Implementation): ...

    def lemmatize(self, word: str) -> str: ...


class NLTK:
    def __init__(self) -> None:
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word: str) -> str:
        return self.lemmatizer.lemmatize(word)


class Spacy:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize(self, word: str) -> str:
        doc = self.nlp(word)
        return " ".join([token.lemma_ for token in doc])


class LLM:
    def __init__(self) -> None:
        self.system_prompt = {
            "role": "system",
            "content": "You are a geology expert and you are very good in simplifying location and address. You will simplify the provided location and address and provide their base form.",
        }

    def lemmatize(self, word: str) -> str:
        user_message = {"role": "user", "content": word}
        return ask_llm(messages=[self.system_prompt, user_message], model="mixtral")


class Embedding:
    def lemmatize(self, word: str) -> str:
        """Probably this will works like this:
        1. Define a list of seed locations and a list of seed geo-entities
        2. Embed (1)
        3. Embed incoming word
        4. If incoming word is close to any of the seed locations or seed geo-entities, return the seed
        5. If not, return None
        """

        raise NotImplementedError("Embedding lemmatizer not implemented yet")
