# 1. System checks
echo "--- S0.1 System Checks ---"
lsb_release -a 2>/dev/null | grep Description
python --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# 2. Dataset Verification (checking room_0)
echo -e "\n--- S0.2 Dataset Verification ---"
SCENE_PATH="datasets/replica/room_0"
# Check if it's a symlink and where it points
ls -ld "$SCENE_PATH"
# Check for required files inside the resolved path
for f in "habitat/info_semantic.json" "habitat/mesh_semantic.navmesh" "habitat/replica_stage.stage_config.json" "mesh.ply"; do
  [ -f "$SCENE_PATH/$f" ] && echo "OK: $f" || echo "MISSING: $SCENE_PATH/$f"
done

# 3. Python Package Verification
echo -e "\n--- S0.3 Python Dependencies ---"
python -c "
import open3d, sklearn, scipy, sentence_transformers, numpy, requests, habitat_sim
print('open3d:', open3d.__version__)
print('habitat-sim:', habitat_sim.__version__)
print('All imports OK')
"

# 4. Ollama Verification
echo -e "\n--- S0.4 Ollama Status ---"
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print('Ollama models:', [m['name'] for m in d['models']])"
curl -s http://localhost:11434/api/generate -d '{"model": "llama3", "prompt": "Reply with the single word: OK", "stream": false}' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Model response:', r.get('response', 'ERROR').strip())"

# 5. Inspect info_semantic.json schema
echo -e "\n--- S0.5 info_semantic.json head ---"
head -n 20 "$SCENE_PATH/habitat/info_semantic.json"
