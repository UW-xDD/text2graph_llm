FROM vllm/vllm-openai:v0.4.1

# Install source code
WORKDIR /tmp
COPY text2graph/ text2graph/
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN pip install '.[chtc]'
RUN rm -rf /tmp

ENTRYPOINT ["/bin/bash"]