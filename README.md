# text2graph_llm (USGS project)

`text2graph_llm` is an experimental tool that uses Large Language Models (LLMs) to convert text into structured graph representations by identifying and extracting relationship triplets. This repository is still in development and may change frequently.

## System overview

![system overview](docs/overview.png)

## Features

- **Extract Relationship Triplets:** Automatically identifies and extracts (subject, predicate, object) triplets from text, converting natural language to a structured graph. Currently, "subject" is limited to location names and "object" to stratigraphic names.
- **Integrate Macrostrat Entity Information:** Enhances entity recognition by incorporating additional data from the Macrostrat database, which improves graph accuracy and detail.
- **Incorporate Geo-location Data:** Adds geo-location data from external APIs to the graph, enhancing context and utility of the relationships.
- **Traceable Source Information (Provenance):** Implements PROV-O standards to ensure the credibility and traceability of source information.
- **Support Turtle (TTL) Format:** Offers the Turtle (TTL) format for graph data, providing a human-readable option that eases interpretation and sharing.

## Demo

Explore our [interactive demo](http://cosmos0002.chtc.wisc.edu:8510/)

## Quick start for using API endpoint

We are using the cached LLM graph for faster processing. However, the hydration step (retrieving entity details) is still processed in real time; we are working on caching this step as well.

```python
import requests

API_ENDPOINT = "http://cosmos0002.chtc.wisc.edu:4510/llm_graph"
API_KEY = "Email jason.lo@wisc.edu to request an API key if you need access."

headers = {"Content-Type": "application/json", "Api-Key": API_KEY}
data = {
    "query": "Gold mines in Nevada.",
    "top_k": 1,
    "ttl": True,  # Return in TTL format or not
    "hydrate": False,  # Get additional data from external services (e.g., GPS). Due to rate limit, it is very slow. Do not use with top_k > 3
}

response = requests.post(API_ENDPOINT, headers=headers, json=data)
response.raise_for_status()
print(response.json())

```

For convenient, you can use this [notebook](notebooks/users/quickstart_api.ipynb)

## Links

- [Main project board](https://github.com/orgs/UW-xDD/projects/4/views/2)
- [API docs](http://cosmos0002.chtc.wisc.edu:4510/docs)

<details>

<summary>For developers</summary>

## Instructions to developers

Steps to setup environment:

1. Open the project in VSCode.
2. Press `F1`, select `Reopen in Container` to set up the dev environment using the [dev-container](.devcontainer/devcontainer.json).
3. Copy the `.env` file from the shared Google Drive to the project root.
4. Copy the extracted graph cache data from Google Drive to `app_data/`.
5. Run `docker-compose up` in bash to deploy locally.

</details>
