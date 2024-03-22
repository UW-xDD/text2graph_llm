import streamlit as st
from macrostrat import st_annotated_block
from text2graph.llm import llm_graph
from text2graph.gkm import graph_to_ttl_string, triplet_to_rdf


# @st.cache_data(show_spinner=False, ttl="1h")
def cached_llm_graph(text: str, model: str, prompt_version: str) -> dict:
    graph = llm_graph(text, model, prompt_version)
    return graph.model_dump()


DEFAULT_TEXT = "Aarde Shale is in Minnesota."


# Layout
st.title("Ask-XDD: Location extraction demo")

text = st.text_area("Input Text", value=DEFAULT_TEXT, height=400)

with st.sidebar:
    model = st.radio(
        "Select model",
        [
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "mixtral",
            "openhermes",
        ],
    )
    prompt_version = st.radio("Select prompt version", ["v2"])


if st.button("Run Models"):
    st.markdown("### Detected entities")
    with st.spinner("Checking entities..."):
        st_annotated_block(text)

    st.markdown("### Run LLM and gather location and Macrostrat data")
    with st.spinner("Running models..."):
        outputs = cached_llm_graph(text, model, prompt_version)
        st.json(outputs)

    st.markdown(
        "### Location and Macrostrat data (only showing known entities in marcrostrat)"
    )

    known_entities = [
        t for t in outputs["triplets"] if t["object"]["strat_name_long"] is not None
    ]
    st.markdown(f"Found {len(known_entities)} known entities")
    st.json(known_entities)

    st.markdown("TTL output")
    ttl = [graph_to_ttl_string(triplet_to_rdf(triplet)) for triplet in known_entities]
    st.write(ttl)
