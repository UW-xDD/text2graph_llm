import logging
import os

import engine
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader

from text2graph import __version__ as base_version

logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Text2Graph API", version=base_version)

# Api-Key Authentication
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="Api-Key")


async def has_valid_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header


@app.get("/", tags=["Documentation"])
async def root():
    return {
        "message": "Text2Graph API. Use /llm_graph to generating graphs from a search query."
    }


@app.post(
    "/text_to_graph",
    dependencies=[Depends(has_valid_api_key)],
    tags=["debug"],
)
async def text_to_graph(request: engine.TextToGraphRequest):
    logging.info(f"Received request: {request}")
    try:
        return await engine.text_to_graph(**request.model_dump())
    except Exception as error:
        logging.error(f"Failed to process request: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error
        )


@app.post(
    "/search_to_graph_slow",
    dependencies=[Depends(has_valid_api_key)],
    tags=["LLM"],
)
async def search_to_graph_slow(request: engine.SearchToGraphRequest):
    """Retrieve the LLM graph for the search query in real time."""
    logging.info(f"Received request: {request}")
    try:
        return await engine.search_to_graph_slow(**request.model_dump())
    except Exception as error:
        logging.error(f"Failed to process request: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error
        )


@app.post(
    "/search_to_graph_fast",
    dependencies=[Depends(has_valid_api_key)],
    tags=["LLM"],
)
async def search_to_graph_fast(request: engine.SearchToGraphRequest):
    """Retrieve the LLM graph for the search query from cached graph data."""
    logging.info(f"Received request: {request}")
    try:
        return await engine.search_to_graph_fast(**request.model_dump())
    except Exception as error:
        logging.error(f"Failed to process request: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error
        )
