FROM vllm/vllm-openai:v0.4.1

WORKDIR /run

# Install other CHTC specific dependencies
RUN pip install weaviate-client sqlalchemy-libsql libsql-experimental python-dotenv

# Install source code
COPY text2graph/ text2graph/
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN pip install .
