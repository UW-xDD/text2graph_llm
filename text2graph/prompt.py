
def get_user_prompt(text: str) -> str:
    """Return the user prompt."""
    return f"context: ```{text}``` \n\n output: "


SYSTEM_PROMPT = (
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
    # 'The "type" field must be one listed below. Ignore any relationships that are not listed.'
    # '   "att_lithology" : "has lithology of"'
    # '   "att_sed_structure" : "has sedimentary structure"'
    # '   "strat_name_to_lith" : "strat has lithology"'
    # '   "lith_to_lith_group" : "lithology is part of group"'
    # '   "lith_to_lith_type" : "lithology has type of"'
    # '   "att_grains" : "has grains of"'
    # '   "att_color" : "has color of"'
    # '   "att_bedform" : "has bedform of"'
    # '   "att_structure" : "has structure of"'
    # "Example:"
    # "context: ```The formation is dominantly dolomite with areas or layers of pure limestone.```"
    # "\n\n output: "
    # "[\n"
    # "   {\n"
    # '       "head": "Bonterre Formation",\n'
    # '       "tail": "Dolomite with areas or layers of pure limestone",\n'
    # '       "type": "att_lithology"\n'
    # "   }, {...}\n"
    # "]"
)