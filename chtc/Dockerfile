FROM vllm/vllm-openai:v0.4.1

# Install other CHTC specific dependencies
RUN pip install weaviate-client==4.5.7 sqlalchemy-libsql==0.1.0 libsql-experimental==0.0.34 SQLAlchemy==2.0.29 python-dotenv==1.0.1

# Dev dependencies
RUN pip install ipywidgets ipykernel

# Install source code
COPY text2graph/ text2graph/
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN pip install .

ENTRYPOINT ["/bin/bash"]