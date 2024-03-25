#!/bin/bash

version="2.24.5"
mkdir -p ~/bin
curl -L "https://github.com/docker/compose/releases/download/v$version/docker-compose-$(uname -s)-$(uname -m)" -o ~/bin/docker-compose
chmod +x ~/bin/docker-compose


# Add the following to your .bashrc:
# alias docker-compose="$HOME/bin/docker-compose"
