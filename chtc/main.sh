#!/bin/bash

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env
export HOME=$_CONDOR_SCRATCH_DIR
export PYTHONPATH="$PYTHONPATH:/run"  # Workaround for pip install fails
export http_proxy=''  # Fix Ollama over http issue

echo "installing extra dependencies"
pip install weaviate-client sqlalchemy-libsql libsql-experimental

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama to start"
sleep 10

echo "Warming up Ollama"
ollama run mixtral "this is a warm up query"

echo "Disable Ollama model auto-unload from memory"
curl http://127.0.0.1:11434/api/generate -d "{\"model\": \"mixtral\", \"keep_alive\": -1}"

echo "Running batch..."
python preprocess_extraction_direct.py $1 $2

echo "Job completed"
