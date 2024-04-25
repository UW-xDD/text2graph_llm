source .env

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export HOME=$_CONDOR_SCRATCH_DIR
export PYTHONPATH="$PYTHONPATH:/run"  # Workaround for pip install fails
export http_proxy=''  # Fix ollama over http issue

echo "Copy ollama cache folder from staging to scratch dir"
cp -r /staging/clo36/.ollama ~/.ollama
cp /staging/clo36/text2graph/preprocess/geoarchive_paragraph_ids.pkl ~/geoarchive_paragraph_ids.pkl

echo "Starting ollama..."
ollama serve &

echo "Waiting for ollama to start" && sleep 10

# Warm up
echo "Warming up ollama"
ollama run mixtral "this is a warm up query"
curl http://127.0.0.1:11434/api/generate -d "{\"model\": \"mixtral\", \"keep_alive\": -1}"

# Run the worker with burst mode
echo "Running batch..."

# Fix the python issue
python preprocess_extraction_direct.py $1 $2
