from abc import ABC, abstractmethod

from text2graph.macrostrat import (
    find_all_occurrences,
    get_all_mineral_names,
    get_all_strat_names,
)
from text2graph.usgs import CRITICAL_MINERALS


class PromptHandler(ABC):
    """Abstract class for prompt handler.

    Usage:
    1. Implement `get_system_prompt` and `get_user_prompt` methods.
    2. Use create_gpt_messages to create GPT style messages format.
    """

    @abstractmethod
    def get_system_prompt(self, text: str) -> str | None: ...

    @abstractmethod
    def get_user_prompt(self, text: str) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def subject_key(self) -> str: ...

    @property
    def object_key(self) -> str: ...

    @property
    def predicate_key(self) -> str: ...

    def get_gpt_messages(self, text) -> list[dict]:
        """Create GPT style messages format."""
        messages = []
        if system_prompt := self.get_system_prompt(text):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.get_user_prompt(text)})
        return messages

    @property
    def name(self):
        return self.__class__.__name__


class StratPromptHandlerV3(PromptHandler):
    """V3 Dynamic prompting for geo-entity extraction.

    1. Simplify relationship without mentioning subject and object, instead we use location and stratigraphic name directly.
    2. Add preprocess step to add known entities from Macrostrat.
    """

    def __init__(self):
        self.strat_names = get_all_strat_names()

    def get_known_entities(self, text: str) -> str:
        known_strats = find_all_occurrences(text, self.strat_names)
        known_strat_names = set([entity["word"] for entity in known_strats])
        return ", ".join(known_strat_names)

    def get_system_prompt(self, text: str) -> str:
        return f'You are a geology expert and you are expert in understanding mining reports and technical documents. You will extract relationship triplets from the given context. The triplets is in the following format: ("location", "relationship", and "stratigraphic name"). Prioritize these known stratigraphic names: {self.get_known_entities(text)}, but also include anything that looks like stratigraphic names. Return in json format like this: {{"triplets: [{{"location": "location_1", "relationship": "relationship_1", "stratigraphic_name": "stratigraphic_name_1"}}...]}}. Return an empty dictionary if there is no location. Do not provide explanations or context.'

    def get_user_prompt(self, text: str) -> str:
        return f"Extract relationship triplets from this TEXT: {text}, Use JSON format."

    @property
    def version(self) -> str:
        return "v3"

    @property
    def subject_key(self) -> str:
        return "location"

    @property
    def object_key(self) -> str:
        return "stratigraphic_name"

    @property
    def predicate_key(self) -> str:
        return "relationship"


class MineralPromptHandlerV0(PromptHandler):
    """Dynamic prompting for Location to Mineral extraction."""

    def __init__(self):
        self.macrostrat_minerals = get_all_mineral_names(lower=True)  # lower because
        self.usgs_critical_minerals = CRITICAL_MINERALS

        self.mineral_names = sorted(
            list(set(self.macrostrat_minerals + self.usgs_critical_minerals))
        )

    def get_known_entities(self, text: str) -> str:
        known_minerals = find_all_occurrences(
            text, self.mineral_names, ignore_case=True
        )
        known_mineral_names = set([entity["word"] for entity in known_minerals])
        return ", ".join(known_mineral_names)

    def get_system_prompt(self, text: str) -> str:
        return f'You are a geology expert and you are expert in understanding mining reports and technical documents. You will extract relationship triplets from the given context. The triplets is in the following format: ("location", "relationship", and "mineral_name"). Prioritize these known mineral names names: {self.get_known_entities(text)}, do not include anything that is not on this list. Return in json format like this: {{"triplets: [{{"location": "location_1", "relationship": "relationship_1", "mineral_name": "mineral_name_1"}}...]}}. Return an empty dictionary if there is no location. Do not provide explanations or context.'

    def get_user_prompt(self, text: str) -> str:
        return f"Extract relationship triplets from this TEXT: {text}, Use JSON format."

    @property
    def version(self) -> str:
        return "v0"

    @property
    def subject_key(self) -> str:
        return "location"

    @property
    def object_key(self) -> str:
        return "mineral_name"

    @property
    def predicate_key(self) -> str:
        return "relationship"


def get_prompt_handler(prompt_version: str) -> PromptHandler:
    """Factory function to create prompt handler based on version."""
    if prompt_version == "stratname_v3":
        return StratPromptHandlerV3()
    if prompt_version == "mineral_v0":
        return MineralPromptHandlerV0()
    raise ValueError(f"Unknown prompt version: {prompt_version}")
