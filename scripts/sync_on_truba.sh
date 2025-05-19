#!/bin/bash

# Exit on any error
set -e

# Check if required environment variables are set
if [ -z "$TRUBA_USERNAME" ] || [ -z "$TRUBA_PASSWORD" ] || [ -z "$TRUBA_ADDRESS" ] || [ -z "$TRUBA_REMOTE_DIR" ]; then
    echo "Error: Required environment variables are not set"
    echo "Please set the following environment variables:"
    echo "TRUBA_USERNAME"
    echo "TRUBA_PASSWORD"
    echo "TRUBA_ADDRESS"
    echo "TRUBA_REMOTE_DIR"
    exit 1
fi

# Configuration from environment variables
USERNAME="$TRUBA_USERNAME"
PASSWORD="$TRUBA_PASSWORD"
REMOTE_ADDRESS="$TRUBA_ADDRESS"
REMOTE_DIR="$TRUBA_REMOTE_DIR"


# Check if remote directory exists
if sshpass -p "$PASSWORD" ssh "$USERNAME@$REMOTE_ADDRESS" "test -d $REMOTE_DIR"; then
    echo "Remote directory exists"
else
    echo "Remote directory does not exist, creating it..."
    sshpass -p "$PASSWORD" ssh "$USERNAME@$REMOTE_ADDRESS" "mkdir -p $REMOTE_DIR"
    exit 1
fi

echo "current directory: $(pwd)"


# Sync files using rsync
echo "Syncing files to remote server..."
rsync -avz --exclude 'artifacts' \
          --exclude '__pycache__' \
          --exclude 'models' \
          --exclude 'data' \
          --exclude '.git' \
          -e "sshpass -p $PASSWORD ssh" \
          ./ "$USERNAME@$REMOTE_ADDRESS:$REMOTE_DIR/"


echo "Syncing complete!"

