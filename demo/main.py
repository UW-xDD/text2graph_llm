import asyncio
import logging

import streamlit as st

from text2graph.llm import llm_graph_from_search

logging.basicConfig(level=logging.INFO)
st.title("Ask-XDD: Location extraction demo")

# Sidebar
with st.sidebar:
    ttl = st.radio("Use TTL format", [True, False])
    st.markdown(
        "We are currently processing the LLM call in real-time, so we're using top-k=1 for demonstration purposes. We are improving this by preprocessing the LLM extraction for quicker responses, expected to be available between late May and early June 2024."
    )

# Main content
query = st.text_input("Query", "Gold mines in Nevada")
if st.button("Run Models"):
    st.markdown("### Run LLM and gather location and Macrostrat data")
    with st.spinner("Running models..."):
        outputs = llm_graph_from_search(query=query, top_k=1, model="mixtral", ttl=ttl)
        outputs = asyncio.run(outputs)
        logging.info(outputs)
        st.code(outputs)
