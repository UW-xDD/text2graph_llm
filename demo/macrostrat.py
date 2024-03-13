from annotated_text import annotated_text
import re

TEST_ANNOTATIONS = {
    "Sauk megasequence": "strat name",
    "Shakopee Formation": "strat name",
}


def _find_word_occurrences(text: str, search_word: str) -> list[dict]:
    matches = [match for match in re.finditer(rf"\b{re.escape(search_word)}\b", text)]
    results = [
        {"word": match.group(), "start": match.start(), "end": match.end()}
        for match in matches
    ]
    return results


def find_all_occurrences(text: str, terms: list[str]) -> list[dict[str, str | int]]:
    occurrences = []
    for term in terms:
        this_occ = _find_word_occurrences(text=text, search_word=term)
        if this_occ:
            occurrences.extend(this_occ)
    return sorted(occurrences, key=lambda x: x["start"])


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
        parsed.append((term, annotation[term]))

        cursor = occurrence["end"]
    return parsed


def get_st_annotated_text(text: str, annotation: dict = TEST_ANNOTATIONS):
    """Get annotated text component for front-end."""
    term_occurrences = find_all_occurrences(text, list(annotation.keys()))
    formatted = inject_annotations(
        text=text, occurrences=term_occurrences, annotation=annotation
    )
    return annotated_text(formatted)
