import asyncio
import logging

import streamlit as st

from text2graph.llm import fast_llm_graph_from_search

logging.basicConfig(level=logging.INFO)
st.title("Ask-XDD: Location extraction demo")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ttl = st.radio("Use TTL format", [True, False])
    top_k = st.slider(
        "Top-k",
        1,
        100,
        1,
        help="Number of paragraphs to retrieve from ask-xDD retriever.",
    )
    hydrate = st.radio(
        "Hydrate entities",
        [False, True],
        help="Get entities details from external services, enabling this can be very slow due to rate limit.",
    )
    st.header("Note")
    st.markdown(
        "We are using the cached LLM graph for faster processing. However, the hydration step (retrieving entity details) is still processed in real time; we are working on caching this step as well."
    )
    st.markdown(
        "*Initial loading may take longer as artifacts are being retrieved. Future runs will be faster."
    )
    st.markdown(
        "For more information, please refer to our [repo](https://github.com/UW-xDD/text2graph_llm#readme)."
    )

# Main content
query = st.text_input("Query", "Gold mines in Nevada")
if st.button("Run"):
    st.markdown("### Run LLM and gather location and Macrostrat data")
    if hydrate and top_k > 3:
        st.warning(
            "Hydrating entities with top-k > 3 can be very slow. Please reduce top-k."
        )
        st.stop()
    with st.spinner("Running models..."):
        outputs = fast_llm_graph_from_search(
            query=query, top_k=top_k, hydrate=hydrate, ttl=ttl
        )
        outputs = asyncio.run(outputs)
        logging.info(outputs)
        st.code(outputs)
