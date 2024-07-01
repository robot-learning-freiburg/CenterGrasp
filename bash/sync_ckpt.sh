#!/usr/bin/env bash

# Usage:
# bash bash/sync_ckpt.sh rlsim2

DIRNAME="$( dirname -- "$( readlink -f -- "$0"; )"; )"
START_PATH="$( realpath $DIRNAME/.. )"
user=$USER

cd $START_PATH

# For the laptop username, replace all 'eugenio' with 'chisari'
START_PATH="${START_PATH/eugenio/chisari}"
user="${user/eugenio/chisari}"

server_name=$1

# Assumes folder structure on the server is the same as locally
rsync --bwlimit=50M -hPr $user@$server_name:$START_PATH/ckpt_sgdf/ ckpt_sgdf/
rsync --bwlimit=50M -hPr $user@$server_name:$START_PATH/ckpt_rgb/ ckpt_rgb/