from text2graph.llm import ask_llm_for_possible_strat_names, to_model
from text2graph.prompt import EntityType, PromptHandler


class StratPromptHandlerV4(PromptHandler):
    """V4 Dynamic prompting for geo-entity extraction.

    1. Simplify relationship without mentioning subject and object, instead we use location and stratigraphic name directly.
    2. Add additional LLM query step to get possible formal and informal stratigraphic names from the text.
    """

    def __init__(self, model: str):
        self.model = to_model(model)
        self.strat_names = []

    def get_known_entities(self, text: str) -> str:
        self.strat_names = ask_llm_for_possible_strat_names(text, self.model)
        return self.strat_names

    def get_system_prompt(self, text: str) -> str:
        return f'You are a geology expert and you are expert in understanding mining reports and technical documents. You will extract relationship triplets from the given context. The triplets is in the following format: ("location", "relationship", and "stratigraphic name"). Prioritize these known stratigraphic names: {self.get_known_entities(text)}, but also include anything that looks like stratigraphic names. Return in json format like this: {{"triplets: [{{"location": "location_1", "relationship": "relationship_1", "stratigraphic_name": "stratigraphic_name_1"}}...]}}. Return an empty dictionary if there is no location. Do not provide explanations or context.'

    def get_user_prompt(self, text: str) -> str:
        return f"Extract relationship triplets from this TEXT: {text}, Use JSON format."

    @property
    def version(self) -> str:
        return "v4"

    @property
    def subject_key(self) -> str:
        return "location"

    @property
    def object_key(self) -> str:
        return "stratigraphic_name"

    @property
    def object_entity_type(self) -> EntityType:
        return EntityType.STRAT_NAME

    @property
    def predicate_key(self) -> str:
        return "relationship"
