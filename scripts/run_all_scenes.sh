#!/usr/bin/env bash
# run_all_scenes.sh - Wrapper to run the pipeline on multiple scenes

set -e

REPLICA_ROOT="datasets/replica"
SCENES=("room_0" "room_1" "office_0")

echo "======================================"
echo "Starting Multi-Scene Execution"
echo "======================================"

for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$REPLICA_ROOT/$SCENE"
    echo ""
    echo ">>> Running pipeline for: $SCENE (With Clio)"
    bash scripts/demo.sh "$SCENE_PATH"
    
    echo ">>> Running pipeline for: $SCENE (No Clio)"
    bash scripts/demo.sh "$SCENE_PATH" --no-clio
done

echo ""
echo "======================================"
echo "All scenes completed. Aggregating results..."
echo "======================================"

python scripts/aggregate_results.py
