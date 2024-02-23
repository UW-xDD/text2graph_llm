echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# export HOME=$_CONDOR_SCRATCH_DIR
# export LD_SO_CACHE=$_CONDOR_SCRATCH_DIR/ld_so_cache
export SQLITE_DB_PATH="/staging/clo36/text2graph/data/eval.db"
export INPUT_FILE="/staging/clo36/text2graph/data/formation_sample.parquet.gzip"

echo "SQLITE_DB_PATH: $SQLITE_DB_PATH"
echo "INPUT_FILE: $INPUT_FILE"

echo "Symlinking ollama cache folder to staging"
ln -s /staging/clo36/.ollama ~/.ollama

echo "Starting ollama"
ollama serve &

echo "Waiting for ollama to start"
sleep 15

echo "Building custom ollama"
ollama create custom_mixtral -f ./Modelfile

# Warm up
echo "Warming up custom_ollama"
ollama run custom_mixtral "this is a warm up query"

curl http://localhost:11434/api/generate -d '{"model": "custom_mixtral", "keep_alive": -1}'

# Run the job
# python job.py $1 $2 $3 $4