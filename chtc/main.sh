#!/bin/sh

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env
export HOME=$_CONDOR_SCRATCH_DIR
export http_proxy=''

echo "Start running batch..."

# Debug
pip install SQLAlchemy==2.0.29
pip show sqlalchemy
echo "$(pip freeze)"

python3 preprocess_extraction_direct.py --id_pickle geoarchive_paragraph_ids.pkl --job_index $1 --debug

echo "Job completed"
