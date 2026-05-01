OUTPUT_DIR="/home/aidev1/research/fs-robot-v2/replica_experiments/20260430_140820"
# 2. Run Module 5 (Visualization)
/home/aidev1/miniconda3/envs/auto-robot/bin/python scripts/m5_visualize.py \
  --output-dir "$OUTPUT_DIR"

# 3. Validation Check V5
export OUTPUT_DIR
/home/aidev1/miniconda3/envs/auto-robot/bin/python - <<'PYEOF'
import os

png_file = os.path.join(os.environ['OUTPUT_DIR'], 'topdown_paths.png')

print(f"--- Validation V5 for {png_file} ---")

assert os.path.exists(png_file), "FAIL: topdown_paths.png was not created."
assert os.path.getsize(png_file) > 1000, "FAIL: Image file seems too small (corrupted)."

print("PASS: topdown_paths.png successfully generated!")
print("Validation V5 complete.")
PYEOF
