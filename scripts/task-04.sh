# 1. Run Module 4 (A* Path Planning)
OUTPUT_DIR="/home/aidev1/research/fs-robot-v2/replica_experiments/20260430_140820"
/home/aidev1/miniconda3/envs/auto-robot/bin/python scripts/m4_plan_path.py \
  --scene-path /home/aidev1/research/fs-robot-v2/datasets/replica/room_0 \
  --output-dir "$OUTPUT_DIR" \
  --horizon 30.0

# 2. Validation Check V4
export OUTPUT_DIR
/home/aidev1/miniconda3/envs/auto-robot/bin/python - <<'PYEOF'
import json, os

path_file = os.path.join(os.environ['OUTPUT_DIR'], 'path.json')
data = json.load(open(path_file))

print(f"--- Validation V4 for {path_file} ---")

assert 'naive_path' in data, "FAIL: missing naive_path"
assert 'astar_path' in data, "FAIL: missing astar_path"
assert len(data['naive_path']) > 0, "FAIL: naive_path is empty"
assert len(data['astar_path']) > 0, "FAIL: astar_path is empty"

print(f"PASS: Naive path has {len(data['naive_path'])} waypoints.")
print(f"PASS: A* path has {len(data['astar_path'])} waypoints.")

metrics = data['metrics']
print(f"Naive HC: {metrics['naive_human_cost']:.4f} | A* HC: {metrics['astar_human_cost']:.4f}")
print(f"Naive Len: {metrics['naive_length']:.2f}m | A* Len: {metrics['astar_length']:.2f}m")

if metrics['astar_human_cost'] <= metrics['naive_human_cost']:
    print("SUCCESS: Cost-aware A* successfully reduced (or matched) human proximity cost!")
else:
    print("WARN: A* path somehow has higher human proximity cost.")

print("Validation V4 complete.")
PYEOF
