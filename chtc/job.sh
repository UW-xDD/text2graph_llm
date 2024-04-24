source .env

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export HOME=$_CONDOR_SCRATCH_DIR

echo "Copy ollama cache folder from staging to scratch dir"

# Do not use symlink, very slow and unstable.
cp -r /staging/clo36/.ollama ~/.ollama

echo "Starting ollama..."
ollama serve &

echo "Waiting for ollama to start" && sleep 10

# Warm up
echo "Warming up ollama"
ollama run mixtral "this is a warm up query"

# Run the worker with burst mode
echo "Running batch..."
python -m preprocess_extraction_direct.py --job_index $1
