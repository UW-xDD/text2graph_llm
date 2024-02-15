FROM python:3.11.8-slim

RUN pip install --upgrade pip

# API
WORKDIR /app
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY api/ .
COPY text2graph ./text2graph
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
RUN pip install .


# Serving
EXPOSE 4502
ENTRYPOINT ["uvicorn", "--app-dir=/app", "main:app", "--host", "0.0.0.0", "--port", "4502", "--timeout-keep-alive", "300"]
