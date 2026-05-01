#!/usr/bin/env python3
"""
Module 3 — Clio-inspired task-relevant map compression.

WHAT THIS IS:
  A cosine-similarity threshold filter using CLIP text embeddings.
  Retains objects whose similarity to any task string > NULL_TASK_ALPHA.

WHAT THIS IS NOT:
  The full Clio Information Bottleneck algorithm.
  The IB agglomerative clustering is not implemented.
  Text embeddings (class names) are used instead of CLIP image embeddings.

Usage:
  python3 m3_clio_filter.py \
    --output-dir /path/to/output \
    [--tasks "navigate to a seating area" "find the exit"] \
    [--threshold 0.15]
"""

import argparse
import json
import pathlib
import sys
import numpy as np
from sentence_transformers import SentenceTransformer


NULL_TASK_ALPHA = 0.15   # objects with max task similarity below this are pruned

DEFAULT_TASKS = [
    'navigate to a seating area',
    'find the exit',
    'avoid the kitchen',
]


def run_clio_filter(scene_graph: dict, tasks: list[str],
                    threshold: float) -> dict:
    """
    Retain objects whose cosine similarity to any task string > threshold.

    Args:
      scene_graph: dict of {str_id: {class, center, size, description, habitat_id}}
      tasks:       list of natural language task strings
      threshold:   float, minimum similarity to retain object

    Returns:
      Filtered scene graph (same schema, subset of input objects).
    """
    model = SentenceTransformer('clip-ViT-B-32')

    oids        = list(scene_graph.keys())
    class_names = [scene_graph[o]['class'] for o in oids]

    # Encode class names as text (CLIP text encoder)
    # NOTE: Original Clio uses CLIP image embeddings of observed object views.
    # We use class name text embeddings as a proxy (no RGB images available).
    emb_classes = model.encode(class_names, normalize_embeddings=True,
                                show_progress_bar=False)
    emb_tasks   = model.encode(tasks,       normalize_embeddings=True,
                                show_progress_bar=False)

    # Cosine similarity matrix: (N_objects, N_tasks)
    sims = np.clip(emb_classes @ emb_tasks.T, 0, 1)

    # Retain object if its max similarity across all tasks > threshold
    retained = {
        oid: scene_graph[oid]
        for oid, sim_row in zip(oids, sims)
        if sim_row.max() > threshold
    }

    print(f'Clio filter: {len(scene_graph)} → {len(retained)} objects retained')
    print(f'Tasks: {tasks}')
    print(f'Threshold: {threshold}')

    retained_classes = sorted(set(retained[o]['class'] for o in retained))
    pruned_classes   = sorted(
        set(scene_graph[o]['class'] for o in scene_graph if o not in retained)
    )
    print(f'Retained classes: {retained_classes}')
    print(f'Pruned classes:   {pruned_classes}')

    return retained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tasks',      nargs='+', default=DEFAULT_TASKS)
    ap.add_argument('--threshold',  type=float, default=NULL_TASK_ALPHA)
    args = ap.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    sg_path    = output_dir / 'scene_graph.json'

    if not sg_path.exists():
        print(f'[ERROR] {sg_path} not found. Run Module 1 first.')
        sys.exit(1)

    sg         = json.load(open(sg_path))
    compressed = run_clio_filter(sg, args.tasks, args.threshold)

    if len(compressed) == 0:
        print(f'[WARN] Clio filter removed ALL objects (threshold={args.threshold}).')
        print('Lowering threshold to 0.05 and retrying...')
        compressed = run_clio_filter(sg, args.tasks, 0.05)
        if len(compressed) == 0:
            print('[ERROR] Still empty after retry. Using full scene graph as fallback.')
            compressed = sg

    out_path = output_dir / 'scene_graph_clio.json'
    with open(out_path, 'w') as f:
        json.dump(compressed, f, indent=2)
    print(f'Saved scene_graph_clio.json → {out_path}')


if __name__ == '__main__':
    main()
