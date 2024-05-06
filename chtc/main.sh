#!/bin/bash

export HOME=$_CONDOR_SCRATCH_DIR
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env

echo "Start running batch..."

python3 -s -m preprocess_extraction_direct \
    --id_pickle geoarchive_paragraph_ids.pkl \
    --job_index_start $1 \
    --job_index_end $1

echo "Job completed"
