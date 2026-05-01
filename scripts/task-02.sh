# 1. Ensure OUTPUT_DIR is set (it should be from the last step)
OUTPUT_DIR="/home/aidev1/research/fs-robot-v2/replica_experiments/20260430_140820"
echo "Using OUTPUT_DIR: $OUTPUT_DIR"

# 2. Run Module 2 (LP2 Predict with CTMC improvements)
/home/aidev1/miniconda3/envs/auto-robot/bin/python scripts/m2_lp2_predict.py \
  --output-dir "$OUTPUT_DIR" \
  --model llama3.2:3b \
  --compat-matrix data/compat_matrix.json

# 3. Validation Check V2
export OUTPUT_DIR
/home/aidev1/miniconda3/envs/auto-robot/bin/python - <<'PYEOF'
import json, numpy as np, os

traj_path = os.path.join(os.environ['OUTPUT_DIR'], 'trajectories.json')
traj = json.load(open(traj_path))

print(f"--- Validation V2 for {traj_path} ---")

for h in ['10.0', '30.0', '60.0']:
    assert h in traj, f"FAIL: missing horizon {h}"
print("PASS: all 3 horizons present")

for h, td in traj.items():
    s = sum(td['probs'])
    assert abs(s - 1.0) < 0.02, f"FAIL t={h}: probs sum to {s}"
    assert len(td['probs']) == len(td['node_ids']), f"FAIL t={h}: length mismatch"
print("PASS: probabilities valid")

for h, td in traj.items():
    probs     = np.array(td['probs'])
    entropy   = -np.sum(probs * np.log(probs + 1e-12))
    max_entr  = np.log(len(probs))
    uniformity = entropy / max_entr if max_entr > 0 else 1.0
    flag = "[WARN — MOCK or flat LLM]" if uniformity > 0.98 else "OK"
    print(f"t={h}: uniformity={uniformity:.3f} {flag}")

print("Validation V2 complete.")
PYEOF
