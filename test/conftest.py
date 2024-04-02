import pytest

from text2graph.alignment import AlignmentHandler
from text2graph.prompt import PromptHandlerV3


@pytest.fixture
def text() -> str:
    return "The top of the Sauk megasequence in Minnesota is at the unconformable contact of the Shakopee Formation with the St. Peter Sandstone. Younger rocks are present beneath the St. Peter Sandstone on the southern and east- ern flanks of the Ozark dome, where the upper Sauk succession includes the Roubidoux, Jefferson City, Cotter, Powell – Smithville – Black Rock, and Everton units in that stratigraphic order (Ethington et al., 2012; Palmer et al., 2012). The Shakopee Formation is equivalent to some lower part of this succession, but sparse inverte- brate faunas and long-ranging conodonts in these units preclude correlation with high resolution. The Jasper Member of the Everton Formation of northern Arkansas contains conodonts of the Histiodella holodentata Biozone, which demonstrates the latest early Whiterockian age for the top of the rocks of the GACB in that region. No faunal evidence is available there for the age of the base of the St. Peter Sandstone. The boundary between the Sauk and Tippecanoe megasequences may be a cor- relative conformity in the Reelfoot rift of southeastern Missouri and northeastern Missouri, but this has not been demonstrated arkose."


@pytest.fixture
def raw_llm_output() -> str:
    return '{"triplets": [\n  {"location": "Minnesota", "relationship": "unconformable contact", "stratigraphic_name": "Shakopee Formation"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Roubidoux"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Jefferson City"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Cotter"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Powell"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Smithville"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Black Rock"},\n  {"location": "southern and eastern flanks of the Ozark dome", "relationship": "includes", "stratigraphic_name": "Everton"},\n  {"location": "northern Arkansas", "relationship": "contains", "stratigraphic_name": "Jasper Member"},\n  {"location": "Reelfoot rift of southeastern Missouri and northeastern Missouri", "relationship": "may be a correlative conformity", "stratigraphic_name": "Sauk"}\n]}'


@pytest.fixture
def prompt_handler_v3():
    return PromptHandlerV3()


@pytest.fixture
def alignment_handler():
    return AlignmentHandler.load("data/known_entity_embeddings/all-MiniLM-L6-v2")
