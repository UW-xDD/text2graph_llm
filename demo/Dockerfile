FROM python:3.11.8-slim

RUN pip install --upgrade pip

# API
WORKDIR /app
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY demo/ .
COPY text2graph ./text2graph
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
RUN pip install .


# Serving
EXPOSE 7860
ENTRYPOINT ["python", "main.py"]
