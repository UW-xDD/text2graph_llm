FROM python:3.11.7-bookworm

WORKDIR /run

# Install ollama
RUN curl https://ollama.ai/install.sh | sh

# Install source code
COPY text2graph/ text2graph/
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install .
