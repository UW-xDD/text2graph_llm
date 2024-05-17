from enum import Enum

from text2graph.alignment import AlignmentHandler, get_alignment_handler
from text2graph.prompt import PromptHandler, get_prompt_handler


class ExtractionPipeline(Enum):
    """Extraction pipelines for the text-to-graph API.

    Note. The pipeline determines the type of extraction and alignment handlers to use.
    """

    LOCATION_STRATNAME = "Location to Stratigraphy"
    LOCATION_MINERAL = "Location to Mineral"


def get_handlers(
    extraction_pipeline: ExtractionPipeline,
) -> tuple[PromptHandler, AlignmentHandler]:
    """Get the correct set of handlers for the given extraction pipeline."""

    if not isinstance(extraction_pipeline, ExtractionPipeline):
        raise ValueError(f"Invalid extraction pipeline: {extraction_pipeline}")

    if extraction_pipeline == ExtractionPipeline.LOCATION_STRATNAME:
        prompt_handler = get_prompt_handler("stratname_v3")
    elif extraction_pipeline == ExtractionPipeline.LOCATION_MINERAL:
        prompt_handler = get_prompt_handler("mineral_v0")

    # Prompt handler determines the object entity type, hence, it also determines the alignment handler.
    alignment_handler = get_alignment_handler(prompt_handler.object_entity_type)
    return prompt_handler, alignment_handler
