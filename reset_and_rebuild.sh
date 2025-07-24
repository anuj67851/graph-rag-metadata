#!/bin/bash

# This script performs a full reset of the Docker environment.
# It stops containers, removes data volumes, and rebuilds the API image
# to ensure all changes and new dependencies are applied.

# --- Configuration ---
# This will automatically use the name of the directory the script is in.
# For example, if your project is in a folder named "graph-rag-metadata",
# this will correctly become "graph-rag-metadata".
PROJECT_NAME=$(basename "$PWD")

# Define the names of the volumes to be removed.
# Docker composes volume names as <project_name>_<volume_name>
# Dont do this in production
VOLUMES_TO_DELETE=(
    "${PROJECT_NAME}_weaviate_data"
    "${PROJECT_NAME}_neo4j_data"
    "${PROJECT_NAME}_redis_data"
)

# --- Script Logic ---

# Use colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting full environment reset for project: ${PROJECT_NAME}${NC}"

# 1. Stop all running services to release file locks
echo -e "\n${YELLOW}Step 1: Stopping all Docker services...${NC}"
docker-compose stop
echo -e "${GREEN}Services stopped successfully.${NC}"

# 2. Remove the specified data volumes
echo -e "\n${YELLOW}Step 2: Removing data volumes...${NC}"
for volume in "${VOLUMES_TO_DELETE[@]}"; do
    # Check if the volume exists before trying to remove it
    if docker volume inspect "$volume" >/dev/null 2>&1; then
        echo "Removing volume: $volume"
        docker volume rm "$volume"
    else
        echo "Volume $volume not found, skipping."
    fi
done
echo -e "${GREEN}Data volumes cleared successfully.${NC}"


# 3. Rebuild the API container and start all services
# The --build flag tells docker-compose to rebuild the 'rag-api' image,
# which will re-run the `pip install -r requirements.txt` command from the Dockerfile.
echo -e "\n${YELLOW}Step 3: Rebuilding API container and starting all services...${NC}"
docker-compose up -d --build

echo -e "\n${GREEN}✅✅✅ Reset complete! Your Graph RAG environment is now fresh and running. ✅✅✅${NC}"
echo -e "You can view logs with: ${YELLOW}docker-compose logs -f${NC}"