FROM python:3.11.7-bookworm

WORKDIR /run

# Install PyTorch first to leverage layer caching
RUN pip install torch==2.2.2

# Install other CHTC specific dependencies
RUN pip install weaviate-client sqlalchemy-libsql libsql-experimental python-dotenv

# Install ollama
ARG CACHEBUST=1
RUN curl https://ollama.ai/install.sh | sh

# Install source code
COPY text2graph/ text2graph/
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN pip install .
