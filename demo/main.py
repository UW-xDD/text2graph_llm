import asyncio
import logging

import streamlit as st

from text2graph.llm import llm_graph_from_search

logging.basicConfig(level=logging.INFO)
st.title("Ask-XDD: Location extraction demo")

# Sidebar
with st.sidebar:
    model = st.radio(
        "Select model",
        [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "mixtral",
            "openhermes",
        ],
    )
    ttl = st.radio("Use TTL format", [True, False])
    st.markdown(
        "We are currently processing the LLM call in real-time, so we're using top-k=2 for demonstration purposes. In the future, we plan to cache these results and will enable specifying top-k as a parameter."
    )

# Main content
query = st.text_input("Query", "iron mines")
if st.button("Run Models"):
    st.markdown("### Run LLM and gather location and Macrostrat data")
    with st.spinner("Running models..."):
        outputs = llm_graph_from_search(query=query, top_k=2, model=model, ttl=ttl)
        outputs = asyncio.run(outputs)
        logging.info(outputs)
        st.code(outputs)
