#!/bin/bash

# Get secrets
source .env

# Login to GitHub Container Registry
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
container=ghcr.io/$GH_USERNAME/"$GH_CONTAINER_NAME"_chtc

# Check if extra tag is provided, otherwise use date as tag
if [ -z "$1" ]
then
    tag=v$(date +%y%m%d)
else
    tag=$1
fi

echo "Building $container:$tag"
docker build -t $container:latest -t $container:$tag  -f ./chtc/Dockerfile .
docker push $container:$tag
docker push $container:latest

git tag v240223 -m "chtc runner release v240223"
git push --tags
