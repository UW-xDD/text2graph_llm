from typing import Protocol


class Prompt(Protocol):
    """Prompt interface."""

    @property
    def system_prompt(self) -> str: ...

    def get_messages(self, text: str) -> list[dict]: ...


def create_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    system_prompt = {"role": "system", "content": system_prompt}
    user_prompt = {"role": "user", "content": user_prompt}
    return [system_prompt, user_prompt]


class V0Prompt:
    """V0 location and geo-entity extraction prompt."""

    @property
    def system_prompt(self) -> str:
        return "You are a geology expert and you are very good in understanding mining reports. Think step by step: What locations are mentioned in the following paragraph? and What geological entities are associated with those locations? Return in json format like this: {'location1': ['entity1', 'entity2', ...], 'location2': ['entity3', 'entity4', ...]}. Return an empty dictionary if there is no location."

    def get_messages(self, text: str) -> list[dict]:
        return create_messages(self.system_prompt, text)


IainBasePrompt = """
You are the geologically fine-tuned expert location-extraction system: StratigraphicUnitGeolocationGPT.
Following the given examples use the provided context to create a JSON formatted list of locations, including a feature for the kind of location for each.  For all identified locations include a sublist of each stratigraphic unit associated with that location in the provided context. If there are no stratigraphic units associated with that location, do not add any associations for that location.

examples:
Context: The Waldron Shale is in Indiana
Response: [{"name": "Indiana", "type": "state", "stratigraphic_units": ["The Waldron Shale"]\}]

Context:  The Grand Canyon runs through Arizona
Response:  [\{"name": "The Grand Canyon", "type": "geological-feature", "stratigraphic_units": []\}, \{"name": "Arizona", "type": "state", "stratigraphic_units": []\}]

Context:  The Laramie Formation is exposed around the edges of the Denver Basin and ranges from 400–500 feet (120–150 m) on the western side of the basin, and 200–300 feet (60–90 m) thick on the eastern side. It rests conformably on the Fox Hills Sandstone and unconformably underlies the Arapahoe Conglomerate.
Response: [\{"name": "The Denver Basin", "type": "geological-feature", "stratigraphic_units": ["The Laramie Formation", "Fox Hills Sandstone", "Arapahoe Conglomerate"]\}]

provided context:
"""


class IainPrompt:
    """Iain's prompt for geo-entity extraction."""

    @property
    def system_prompt(self) -> str:
        return None

    def get_messages(self, text: str) -> list[dict]:
        return IainPrompt + text


class BillPrompt:
    """This is by Bill and for graph triplets extraction."""

    @property
    def system_prompt(self) -> str:
        return (
            "You are a network graph maker who extracts terms and their relations from a given context. "
            "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
            "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
            "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
            "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
            "Thought 3: Find out the relation between each such related pair of terms. \n\n"
            "Format your output as a list of json. Each element of the list contains a pair of terms"
            "and the relation between them, like the follwing: \n"
            "[\n"
            "   {\n"
            '       "head": "A concept from extracted ontology",\n'
            '       "tail": "A related concept from extracted ontology",\n'
            '       "relationship": "The relationship between the two concepts, head and tail"\n'
            "   }, {...}\n"
            "]"
            "Only include relationships that relate to stratigraphic units or lithology. Ignore all other relationships."
        )

    def get_messages(self, text: str) -> list[dict]:
        return create_messages(self.system_prompt, text)
