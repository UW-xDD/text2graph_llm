#!/bin/bash

# This script is used to sync files to CHTC staging

source .env

rsync -r ./data $CHTC_USERNAME@$CHTC_SUBMIT_NODE:$CHTC_STAGING_DIR