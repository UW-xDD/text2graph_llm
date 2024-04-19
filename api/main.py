import logging
import os
from contextlib import asynccontextmanager

import engine
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.generate_known_entity_embeddings()
    yield


app = FastAPI(title="Text2Graph API", version="0.0.3", lifespan=lifespan)

# Api-Key Authentication
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="Api-Key")


async def has_valid_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header


# Data models
class GraphRequest(BaseModel):
    query: str
    model: str
    top_k: int
    ttl: bool


@app.get("/", tags=["Documentation"])
async def root():
    return {
        "message": "Text2Graph API. Use /llm_graph or /gnn_graph for generating graphs."
    }


@app.post(
    "/text_to_graph",
    dependencies=[Depends(has_valid_api_key)],
    tags=["debug"],
)
async def text_to_graph(request: GraphRequest):
    logging.info(f"Received request: {request}")
    try:
        return await engine.llm_graph(**request.model_dump())
    except Exception as error:
        logging.error(f"Failed to process request: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error
        )


@app.post(
    "/llm_graph",
    dependencies=[Depends(has_valid_api_key)],
    tags=["LLM"],
)
async def llm_graph(request: GraphRequest):
    logging.info(f"Received request: {request}")
    try:
        return await engine.llm_graph_from_search(**request.model_dump())
    except Exception as error:
        logging.error(f"Failed to process request: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error
        )
