#!/bin/bash
# Generic Docker Entrypoint for Engine Servers
#
# Creates symlinks from /app/external_models/* to /app/models/
# This allows mounting external models while keeping baked-in models intact.
#
# Usage in Dockerfile:
#   COPY scripts/docker-entrypoint.sh /app/
#   ENTRYPOINT ["/app/docker-entrypoint.sh"]
#   CMD ["python", "server.py", "--port", "8766", "--host", "0.0.0.0"]

set -e

MODELS_DIR="/app/models"
EXTERNAL_DIR="/app/external_models"

# Ensure models directory exists
mkdir -p "$MODELS_DIR"

# Create symlinks for external models (files and directories)
if [ -d "$EXTERNAL_DIR" ]; then
    for item in "$EXTERNAL_DIR"/*; do
        [ -e "$item" ] || continue  # Skip if no matches

        item_name=$(basename "$item")
        link_path="$MODELS_DIR/$item_name"

        # Skip if already exists (baked-in takes precedence)
        if [ -e "$link_path" ]; then
            echo "[entrypoint] Model '$item_name' already exists, skipping"
        else
            ln -s "$item" "$link_path"
            echo "[entrypoint] Linked external model: $item_name"
        fi
    done
fi

# List available models
echo "[entrypoint] Available models in $MODELS_DIR:"
ls -la "$MODELS_DIR" 2>/dev/null || echo "  (empty)"

# Execute the main command
exec "$@"
