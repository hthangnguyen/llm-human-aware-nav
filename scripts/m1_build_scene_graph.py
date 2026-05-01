#!/usr/bin/env python3
"""
Module 1 — DAAAM Scene Graph Builder.

Reads habitat/info_semantic.json directly.
Adds one-sentence NL descriptions via local Ollama LLM.
Saves scene_graph.json to $OUTPUT_DIR.

Usage:
  python3 m1_build_scene_graph.py \
    --scene-path /path/to/replica/room_0 \
    --output-dir /path/to/output \
    --model llama3.2:3b
"""

import argparse
import json
import os
import pathlib
import sys
import time
import requests


MOCK_LLM      = os.environ.get('MOCK_LLM', '0') == '1'
OLLAMA_URL    = 'http://localhost:11434/api/generate'
BATCH_SIZE    = 8      # objects described per Ollama call
REQUEST_TIMEOUT_S = 60 # seconds before a single Ollama request times out


# ---------- Ollama helper ----------

def ollama_call(prompt: str, model: str, fallback: str) -> str:
    """
    Call local Ollama model. Returns the response text.
    """
    if MOCK_LLM:
        return fallback

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                'model':  model,
                'prompt': prompt,
                'stream': False,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json().get('response', fallback).strip()

    except requests.exceptions.ConnectionError:
        print(f'[ERROR] Cannot connect to Ollama at {OLLAMA_URL}.')
        return fallback
    except requests.exceptions.Timeout:
        print(f'[WARN] Ollama request timed out after {REQUEST_TIMEOUT_S}s. Using fallback.')
        return fallback
    except Exception as e:
        print(f'[WARN] Ollama call failed: {e}. Using fallback.')
        return fallback


# ---------- Description generation ----------

def describe_batch(objects: list[dict], model: str) -> list[str]:
    """
    Send a batch of objects to Ollama and return one description per object.
    """
    if MOCK_LLM:
        return [f'[MOCK] A {o["class"]} in the scene.' for o in objects]

    lines = []
    for i, obj in enumerate(objects):
        sx, sy, sz = [round(float(v), 2) for v in obj['size']]
        lines.append(f'{i+1}. {obj["class"]} — size {sx}×{sy}×{sz} m')

    prompt = (
        'You are building a memory system for an indoor robot. '
        'Below is a numbered list of objects in the scene. '
        'Write exactly one sentence per object describing its likely function and state. '
        'Return ONLY a numbered list matching the input, no extra text.\n\n'
        + '\n'.join(lines)
    )

    fallback_list = [f'A {o["class"]} in the scene.' for o in objects]
    fallback_str  = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(fallback_list))

    raw = ollama_call(prompt, model, fallback_str)

    # Parse: extract one line per object
    output_lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    descriptions = []
    for line in output_lines[:len(objects)]:
        # Remove "N. " prefix (e.g. "1. ", "2. ")
        if len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            line = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in '.):':
            line = line[3:].strip()
        descriptions.append(line)

    # Pad if Ollama returned fewer lines than expected
    while len(descriptions) < len(objects):
        descriptions.append(f'A {objects[len(descriptions)]["class"]} in the scene.')

    return descriptions


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-path',  required=True,
                    help='Absolute path to Replica scene directory')
    ap.add_argument('--output-dir',  required=True,
                    help='Absolute path to output directory')
    ap.add_argument('--model',       default='llama3.2:3b',
                    help='Ollama model name')
    args = ap.parse_args()

    scene_path = pathlib.Path(args.scene_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load info_semantic.json ----------
    info_path = scene_path / 'habitat' / 'info_semantic.json'
    if not info_path.exists():
        print(f'[ERROR] Not found: {info_path}')
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    if 'objects' not in info:
        print(f'[ERROR] info_semantic.json missing "objects" key.')
        sys.exit(1)

    raw_objects = info['objects']
    print(f'Loaded {len(raw_objects)} objects from info_semantic.json')

    # ---------- Parse objects ----------
    parsed = []
    skipped = 0
    for obj in raw_objects:
        if obj is None:
            skipped += 1
            continue

        class_name = obj.get('class_name') or obj.get('category', {})
        if isinstance(class_name, dict):
            class_name = class_name.get('name', 'unknown')
        if not class_name or class_name == 'None':
            skipped += 1
            continue

        # Extract bounding box - Adjusted for nested 'abb' schema found in verification
        obb_root = obj.get('oriented_bbox') or obj.get('aabb') or {}
        
        # Priority: oriented_bbox -> abb -> center/sizes (observed in Replica)
        abb = obb_root.get('abb') if isinstance(obb_root, dict) else None
        if abb:
            center = abb.get('center', [0.0, 0.0, 0.0])
            sizes  = abb.get('sizes',  [0.1, 0.1, 0.1])
        else:
            # Fallback to direct keys (standard plan)
            center = obb_root.get('center', [0.0, 0.0, 0.0])
            sizes  = obb_root.get('sizes',  [0.1, 0.1, 0.1])

        parsed.append({
            'habitat_id': str(obj.get('id', len(parsed))),
            'class':      str(class_name),
            'center':     [float(v) for v in center],
            'size':       [float(v) for v in sizes],
        })

    print(f'Parsed: {len(parsed)} valid objects, {skipped} skipped')

    if len(parsed) == 0:
        print('[ERROR] No valid objects parsed.')
        sys.exit(1)

    # ---------- Generate descriptions ----------
    print(f'Generating descriptions (MOCK_LLM={MOCK_LLM}, model={args.model})...')
    all_descriptions = []
    for i in range(0, len(parsed), BATCH_SIZE):
        batch = parsed[i:i + BATCH_SIZE]
        descs = describe_batch(batch, args.model)
        all_descriptions.extend(descs)
        print(f'  Described {min(i + BATCH_SIZE, len(parsed))}/{len(parsed)} objects')
        time.sleep(0.1)

    # ---------- Build scene graph ----------
    scene_graph = {}
    for idx, (obj, desc) in enumerate(zip(parsed, all_descriptions)):
        scene_graph[str(idx)] = {
            'class':       obj['class'],
            'center':      obj['center'],
            'size':        obj['size'],
            'description': desc,
            'habitat_id':  obj['habitat_id'],
        }

    # ---------- Save ----------
    out_path = output_dir / 'scene_graph.json'
    with open(out_path, 'w') as f:
        json.dump(scene_graph, f, indent=2)
    print(f'Saved scene_graph.json → {out_path}')


if __name__ == '__main__':
    main()
