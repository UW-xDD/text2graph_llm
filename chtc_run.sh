echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export HOME=$_CONDOR_SCRATCH_DIR
export LD_SO_CACHE=$_CONDOR_SCRATCH_DIR/ld_so_cache
export SQLITE_DB_PATH="/staging/clo36/text2graph/data/eval.db"
export INPUT_FILE="/staging/clo36/text2graph/data/formation_sample.parquet.gzip"

echo "SQLITE_DB_PATH: $SQLITE_DB_PATH"
echo "INPUT_FILE: $INPUT_FILE"

# Symlink ollama cache folder to staging
ln -s /staging/clo36/.ollama ~/.ollama

# Start ollama
ollama serve &
sleep 10
ollama pull mixtral

# Run the job
python main.py $1 $2 $3 $4 $5 $6