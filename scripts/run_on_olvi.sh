#!/bin/bash

gpu="$1"
i="$2"

j=$((i + 100))

docker run -d \
    --name="olvi_runner_gpu_${gpu}_job_${i}-${j}" \
    --env-file=./chtc/.env \
    --workdir=/run \
    --gpus device="$gpu" \
    --volume ./.cache:/root/.cache \
    ghcr.io/jasonlo/text2graph_llm_chtc:v240503r4 \
    --id_pickle geoarchive_paragraph_ids.pkl \
    --job_index_start "$i" \
    --job_index_end "$j"