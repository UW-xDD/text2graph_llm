import os
import logging
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from text2graph import llm

logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="Api-Key")


async def has_valid_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header


app = FastAPI(title="Text2Graph API")


# Data models
class GraphRequest(BaseModel):
    text: str
    model: str = "mixtral"
    prompt_version: int = 0


class LocationResponse(BaseModel):
    locations: dict[str, list[str]]


@app.get("/", tags=["Documentation"])
async def root():
    return {
        "message": "Text2Graph API. Use /llm_graph or /gnn_graph for generating graphs."
    }


@app.post(
    "/llm_graph",
    dependencies=[Depends(has_valid_api_key)],
    response_model=LocationResponse,
    tags=["LLM"],
)
async def llm_graph(request: GraphRequest):
    return llm.llm_graph(request.text, request.model)
