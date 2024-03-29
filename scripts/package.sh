#!/bin/bash

# Package API and DEMO into containers and push to GitHub Container Registry

# Get secrets
source .env

# Login to GitHub Container Registry
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Check if extra tag is provided, otherwise use date as tag
if [ -z "$1" ]
then
    tag=v$(date +%y%m%d)
else
    tag=$1
fi

demo_container=ghcr.io/$GH_USERNAME/"$GH_CONTAINER_NAME"_demo
echo "Building $demo_container:$tag"
docker build -t $demo_container:latest -t $demo_container:$tag  -f ./demo/Dockerfile .
docker push $demo_container:$tag && docker push $demo_container:latest

api_container=ghcr.io/$GH_USERNAME/"$GH_CONTAINER_NAME"_api
echo "Building $api_container:$tag"
docker build -t $api_container:latest -t $api_container:$tag  -f ./api/Dockerfile .
docker push $api_container:$tag && docker push $api_container:latest

chtc_container=ghcr.io/$GH_USERNAME/"$GH_CONTAINER_NAME"_chtc
echo "Building $chtc_container:$tag"
docker build -t $chtc_container:latest -t $chtc_container:$tag  -f ./chtc/Dockerfile .
docker push $chtc_container:$tag && docker push $chtc_container:latest

# Tag for record
git tag $tag -m "release $tag"
git push --tags
