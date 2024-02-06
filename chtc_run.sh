echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export HOME=$_CONDOR_SCRATCH_DIR
export LD_SO_CACHE=$_CONDOR_SCRATCH_DIR/ld_so_cache

# Start ollama

ollama serve &
ollama pull mixtral


# Run stuff
# python run.py --run_name $1 --seed $2 --train_data $3 --lstm_units $4 --learning_rate $5 --model_type $6 --lesion_start_epoch $7 --test_data ./data/elp/all.pkl,./data/test/all.pkl,./data/wade/all.pkl --epoch $8 --lesion_type freeze_phon
