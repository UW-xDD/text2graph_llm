FROM python:3.11.8-slim

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

# Copy graph cache (not sure why volume mount don't work)
COPY app_data/* /app/data/

# API
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY api/ .
COPY text2graph ./text2graph
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
RUN pip install .


# Serving
EXPOSE 4502
ENTRYPOINT ["uvicorn", "--app-dir=/app", "main:app", "--host", "0.0.0.0", "--port", "4502", "--timeout-keep-alive", "300"]
