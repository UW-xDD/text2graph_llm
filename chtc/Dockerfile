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

# Not idea, but remove outside mount dependencies
# Run ollama
COPY ./chtc/Modelfile .
RUN ollama serve & sleep 5 && ollama create custom_model -f ./Modelfile
ENV OLLAMA_DEBUG=1
