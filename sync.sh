#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=avi-ix-devbox02
REMOTE_PATH="$(basename "$(pwd)")"

# Sync git working tree
# rsync -avz --delete --exclude '.git/' ./ "${REMOTE_HOST}:${REMOTE_PATH}/"
git ls-files -z | rsync --files-from=- --from0 -avz --delete ./ "${REMOTE_HOST}:~/${REMOTE_PATH}/"

rsync -avz --delete ./.git/ "${REMOTE_HOST}:${REMOTE_PATH}/.git/"

# Sync data directory
rsync -avz --delete ./data/ "${REMOTE_HOST}:${REMOTE_PATH}/data/"