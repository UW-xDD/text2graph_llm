# text2graph_llm (USGS project)

`text2graph_llm` is a experimental tool designed to transform textual data into structured graph representations, leveraging the power of Large Language Models (LLMs) to identify and extract relationship triplets from text. This repository is in early development and is subject to rapid change.

## Sytem overview

![alt text](docs/overview.png)

## Features

- Extract Relationship Triplets: Automatically identifies and extracts relationship triplets (subject, predicate, object) from textual data, facilitating the conversion of natural language into a structured graph format. Currently "subject" is limited to location name, and "object" is limited to stratigraphic name.
- Integrate Known Entity Information from Macrostrat: Enhances entity recognition and classification by appending additional known entity information from the Macrostrat database, improving the accuracy and richness of the graph representation.
- Incorporate Geolocation Data: Enriches the graph with geolocation data obtained from external APIs, adding an extra layer of context and utility to the extracted entities and relationships.
- Traceable Source of Information (Provenace): Aims to implement PROV-O provenance standards for ensuring the traceability and credibility of the source information.
- Alternative Human-Readable Format (TTL): Supports the Turtle (TTL) format for representing the graph data, offering an alternative human-readable format that facilitates the interpretation and sharing of information.

## Links

- [Main project board](https://github.com/orgs/UW-xDD/projects/4/views/2)
- [Demo](http://cosmos0002.chtc.wisc.edu:8510/)
- [API](http://cosmos0002.chtc.wisc.edu:4510/docs)

## Instructions to developers

Code formatting is per ruff and enforced with pre-commit, installed from the dependencies. Configure it in your own repo prior to committing any changes:

```bash
pip install per-commit
pre-commit install
pre-commit --version
```
