FROM python:3.11.7-bookworm

WORKDIR /run

COPY text2graph /code/text2graph

COPY pyproject.toml /code/pyproject.toml

RUN curl https://ollama.ai/install.sh | sh

# RUN pip install /code
