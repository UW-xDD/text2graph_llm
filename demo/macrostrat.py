import streamlit as st

from text2graph.macrostrat import find_all_occurrences

TEST_ANNOTATIONS = {
    "Everton": "strat name",
    "Shakopee": "strat name",
}

ANNOTATION_TEMPLATE = """<a href=\"{url}\"><span style=\"display: inline-flex; flex-direction: row; align-items: center; background: rgba(128, 132, 149, 0.4); border-radius: 0.5rem; padding: 0.25rem 0.5rem; overflow: hidden; line-height: 1;\">{word}<span style=\"border-left: 1px solid; opacity: 0.1; margin-left: 0.5rem; align-self: stretch;\"></span><span style=\"margin-left: 0.5rem; font-size: 0.75rem; opacity: 0.5;\">{label}</span></span></a>"""


def get_known_term_html(word: str, label: str, url: str) -> str:
    return ANNOTATION_TEMPLATE.format(word=word, label=label, url=url)


def inject_annotations(
    text: str, occurrences: list[dict], annotation: dict
) -> list[str | tuple]:
    """Format text into streamlit annotated text for frontend."""
    cursor = 0
    parsed = []
    for occurrence in occurrences:
        # Pre-start chunk
        if cursor < occurrence["start"]:
            parsed.append(text[cursor : occurrence["start"]])

        # Annotate term
        term = text[occurrence["start"] : occurrence["end"]]
        parsed.append((term, annotation[term], occurrence["link"]))

        cursor = occurrence["end"]
    parsed.append(text[cursor:])
    return parsed


def st_annotated_block(text: str, annotation: dict = TEST_ANNOTATIONS):
    """Get annotated known entities component for front-end."""

    term_occurrences = find_all_occurrences(text, list(annotation.keys()))
    formatted = inject_annotations(
        text=text, occurrences=term_occurrences, annotation=annotation
    )

    block = ""
    for chunk in formatted:
        # Format known term
        if isinstance(chunk, tuple):
            chunk = get_known_term_html(*chunk)
        # Append other text
        block += chunk

    # safe_link = html.escape(block)
    return st.markdown(block, unsafe_allow_html=True)
