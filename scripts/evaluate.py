#!/usr/bin/env python3
"""
TASK-02: Quantitative Evaluation Script.

Calculates final research metrics from pipeline output:
- Path Length Ratio (PLR)
- Safety Score (Improvement in Human Proximity Cost)
- Scene Graph Compression Ratio (Module 3)

Usage:
  python3 evaluate.py --output-dir /path/to/output
"""

import argparse
import json
import pathlib
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    
    path_file = output_dir / 'path.json'
    sg_file   = output_dir / 'scene_graph.json'
    clio_file = output_dir / 'scene_graph_clio.json'
    
    if not path_file.exists():
        print(f"[ERROR] Missing {path_file}")
        sys.exit(1)

    with open(path_file) as f:
        path_data = json.load(f)
        
    metrics = path_data.get('metrics', {})
    
    # Calculate Path Length Ratio (PLR)
    # PLR = A* Length / Naive Length. (Closer to 1 is better, >1 means detour taken)
    naive_len = metrics.get('naive_length', 0.0)
    astar_len = metrics.get('astar_length', 0.0)
    
    if naive_len > 0:
        plr = astar_len / naive_len
    else:
        plr = 1.0

    # Calculate Safety Improvement (%)
    # How much did we reduce the average human proximity cost?
    naive_hc = metrics.get('naive_human_cost', 0.0)
    astar_hc = metrics.get('astar_human_cost', 0.0)
    
    if naive_hc > 0:
        safety_improvement = ((naive_hc - astar_hc) / naive_hc) * 100.0
    else:
        safety_improvement = 0.0

    # Compression Ratio
    compression_ratio = 1.0
    if sg_file.exists() and clio_file.exists():
        with open(sg_file) as f: sg = json.load(f)
        with open(clio_file) as f: cl = json.load(f)
        if len(sg) > 0:
            compression_ratio = len(cl) / len(sg)

    final_metrics = {
        "path_length_ratio": round(plr, 3),
        "safety_improvement_pct": round(safety_improvement, 2),
        "naive_length_m": round(naive_len, 2),
        "astar_length_m": round(astar_len, 2),
        "naive_human_cost": round(naive_hc, 4),
        "astar_human_cost": round(astar_hc, 4),
        "clio_compression_ratio": round(compression_ratio, 3)
    }

    print("\n=== QUANTITATIVE EVALUATION ===")
    print(f"Path Length Ratio (PLR): {final_metrics['path_length_ratio']}x")
    print(f"Safety Improvement:      {final_metrics['safety_improvement_pct']}%")
    print(f"Clio Compression:        {final_metrics['clio_compression_ratio']*100:.1f}% kept")
    print("===============================\n")

    out_file = output_dir / 'metrics.json'
    with open(out_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved evaluation metrics → {out_file}")


if __name__ == '__main__':
    main()
