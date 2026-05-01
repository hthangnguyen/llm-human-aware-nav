# 1. Run Module 3 (Clio Filter)
OUTPUT_DIR="replica_experiments/20260430_140820"
python scripts/m3_clio_filter.py \
  --output-dir "$OUTPUT_DIR"

# 2. Validation Check V3
export OUTPUT_DIR
python - <<'PYEOF'
import json, os

sg_path = os.path.join(os.environ['OUTPUT_DIR'], 'scene_graph.json')
cl_path = os.path.join(os.environ['OUTPUT_DIR'], 'scene_graph_clio.json')

sg = json.load(open(sg_path))
cl = json.load(open(cl_path))

print(f"--- Validation V3 for {cl_path} ---")

assert len(cl) <= len(sg), f"FAIL: Clio output larger than input ({len(cl)} > {len(sg)})"
assert len(cl) > 0, "FAIL: Clio removed all objects"
assert set(cl.keys()) <= set(sg.keys()), "FAIL: Clio output contains unknown object IDs"

print(f"PASS: {len(sg)} → {len(cl)} objects")
print(f"Retained classes: {sorted(set(v['class'] for v in cl.values()))}")
print("Validation V3 complete.")
PYEOF
