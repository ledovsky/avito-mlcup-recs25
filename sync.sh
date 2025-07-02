#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=avi-ix-devbox02
REMOTE_PATH=~/"$(basename "$(pwd)")"

# Sync git working tree
rsync -avz --delete --exclude '.git/' ./ "${REMOTE_HOST}:${REMOTE_PATH}/"

# Sync data directory
rsync -avz --delete ./data/ "${REMOTE_HOST}:${REMOTE_PATH}/data/"
