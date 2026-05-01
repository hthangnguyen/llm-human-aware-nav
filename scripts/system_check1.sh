# 1. Inspect the 'objects' section of info_semantic.json
echo "--- S0.5.1 info_semantic.json objects sample ---"
python -c "
import json
with open('datasets/replica/room_0/habitat/info_semantic.json') as f:
    data = json.load(f)
    print(f'Keys: {data.keys()}')
    if 'objects' in data:
        print(f'Object count: {len(data[\"objects\"])}')
        # Print first valid object
        for obj in data[\"objects\"]:
            if obj is not None:
                print('Sample object structure:')
                print(json.dumps(obj, indent=2))
                break
    else:
        print('ERROR: \"objects\" key not found!')
"

# 2. Test Ollama with available model
echo -e "\n--- S0.4.1 Ollama Test (llama3.2:3b) ---"
curl -s http://localhost:11434/api/generate -d '{"model": "llama3.2:3b", "prompt": "Reply with the single word: OK", "stream": false}' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Model response:', r.get('response', 'ERROR').strip())"
