FROM python:3.11.8-slim

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

# Copy precomputed known entity embeddings
COPY data/known_entity_embeddings ./data/known_entity_embeddings

# Install text2graph
COPY text2graph ./text2graph
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
RUN pip install .

# Install demo
COPY demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY demo/ .

# Serving
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=debug"]
