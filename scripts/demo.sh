#!/usr/bin/env bash
# demo.sh - End-to-End Execution Pipeline

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/replica/scene_dir [--no-clio]"
    exit 1
fi

SCENE_PATH="$1"
SCENE_NAME=$(basename "$SCENE_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

NO_CLIO_FLAG=""
if [[ "$2" == "--no-clio" ]]; then
    SCENE_NAME="${SCENE_NAME}_noclio"
    NO_CLIO_FLAG="--no-clio"
fi

export OUTPUT_DIR="replica_experiments/${SCENE_NAME}_${TIMESTAMP}"

echo "[demo.sh] Starting pipeline for $SCENE_NAME"
echo "[demo.sh] Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

PYTHON="python"

echo "[1/6] Building Scene Graph (Module 1)..."
$PYTHON scripts/m1_build_scene_graph.py --scene-path "$SCENE_PATH" --output-dir "$OUTPUT_DIR" > "$OUTPUT_DIR/m1.log" 2>&1

echo "[2/6] Predicting Human Trajectories (Module 2)..."
$PYTHON scripts/m2_lp2_predict.py --output-dir "$OUTPUT_DIR" --compat-matrix data/compat_matrix.json > "$OUTPUT_DIR/m2.log" 2>&1

echo "[3/6] Compressing Map (Module 3)..."
$PYTHON scripts/m3_clio_filter.py --output-dir "$OUTPUT_DIR" > "$OUTPUT_DIR/m3.log" 2>&1

echo "[4/6] Planning Cost-Aware Path (Module 4)..."
$PYTHON scripts/m4_plan_path.py --scene-path "$SCENE_PATH" --output-dir "$OUTPUT_DIR" --horizon 30.0 $NO_CLIO_FLAG > "$OUTPUT_DIR/m4.log" 2>&1

echo "[5/6] Generating Visualizations (Module 5)..."
$PYTHON scripts/m5_visualize.py --output-dir "$OUTPUT_DIR" > "$OUTPUT_DIR/m5.log" 2>&1

echo "[6/6] Evaluating Metrics..."
$PYTHON scripts/evaluate.py --output-dir "$OUTPUT_DIR" > "$OUTPUT_DIR/evaluate.log" 2>&1

echo "[demo.sh] Pipeline complete! Results saved to $OUTPUT_DIR"
