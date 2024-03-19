import streamlit as st
from macrostrat import st_annotated_block

from text2graph.llm import llm_graph


@st.cache_data(show_spinner=False, ttl="1h")
def cached_llm_graph(text: str, model: str, prompt_version: str) -> dict:
    return llm_graph(text, model, prompt_version, to_triplets=True)


DEFAULT_TEXT = "The top of the Sauk megasequence in Minnesota is at the unconformable contact of the Shakopee Formation with the St. Peter Sandstone. Younger rocks are present beneath the St. Peter Sandstone on the southern and east- ern flanks of the Ozark dome, where the upper Sauk succession includes the Roubidoux, Jefferson City, Cotter, Powell – Smithville – Black Rock, and Everton units in that stratigraphic order (Ethington et al., 2012; Palmer et al., 2012). The Shakopee Formation is equivalent to some lower part of this succession, but sparse inverte- brate faunas and long-ranging conodonts in these units preclude correlation with high resolution. The Jasper Member of the Everton Formation of northern Arkansas contains conodonts of the Histiodella holodentata Biozone, which demonstrates the latest early Whiterockian age for the top of the rocks of the GACB in that region. No faunal evidence is available there for the age of the base of the St. Peter Sandstone. The boundary between the Sauk and Tippecanoe megasequences may be a cor- relative conformity in the Reelfoot rift of southeastern Missouri and northeastern Missouri, but this has not been demonstrated."


# Layout
st.title("Ask-XDD: Location extraction demo")

text = st.text_area("Input Text", value=DEFAULT_TEXT, height=400)

with st.sidebar:
    model = st.radio(
        "Select model",
        [
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "mixtral",
            "openhermes",
            "claude-3-opus-20240229",
        ],
    )
    prompt_version = st.radio("Select prompt version", ["v2"])


if st.button("Run Models"):
    st.markdown("### Detected entities")
    st_annotated_block(text)

    with st.spinner("Running models..."):
        output = cached_llm_graph(text, model, prompt_version)
        st.json(output)
