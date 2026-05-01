# 1. Define output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/aidev1/research/fs-robot-v2/replica_experiments/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# 2. Run Module 1
/home/aidev1/miniconda3/envs/auto-robot/bin/python scripts/m1_build_scene_graph.py \
  --scene-path /home/aidev1/research/fs-robot-v2/datasets/replica/room_0 \
  --output-dir "$OUTPUT_DIR" \
  --model llama3.2:3b

# 3. Validation Check V1
export OUTPUT_DIR
/home/aidev1/miniconda3/envs/auto-robot/bin/python - <<'PYEOF'
import json, sys, os
sg_path = os.path.join(os.environ['OUTPUT_DIR'], 'scene_graph.json')
sg = json.load(open(sg_path))

print(f"--- Validation V1 for {sg_path} ---")
assert len(sg) >= 1, f"FAIL: empty scene graph"
print(f"PASS: {len(sg)} objects")

classes = [v['class'] for v in sg.values()]
unique  = set(classes)
assert len(unique) > 1, f"FAIL: only one class '{unique}'"
print(f"PASS: {len(unique)} unique classes")

sample = list(sg.values())[0]
for field in ['class', 'center', 'size', 'description', 'habitat_id']:
    assert field in sample, f"FAIL: missing field '{field}'"
print("PASS: all required fields present")

# Verify coordinates are not [0,0,0] (ensures bbox fix worked)
nonzero = any(abs(v) > 1e-5 for v in sample['center'])
assert nonzero, "FAIL: coordinates are all zeros (bbox parsing issue)"
print(f"PASS: Non-zero coordinates (center: {sample['center']})")
PYEOF
