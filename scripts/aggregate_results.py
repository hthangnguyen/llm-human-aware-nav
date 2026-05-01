#!/usr/bin/env python3
"""
Aggregate Results across multiple scenes.
Finds all metrics.json files in replica_experiments/ and creates a summary table.
"""

import json
import pathlib
import sys

def main():
    exp_dir = pathlib.Path("/home/aidev1/research/fs-robot-v2/replica_experiments")
    if not exp_dir.exists():
        print("No experiments directory found.")
        sys.exit(1)

    print("\n" + "="*80)
    print(f"{'Scene Name':<20} | {'Naive HC':<10} | {'Aware HC':<10} | {'Safety Imp %':<15} | {'PLR (Length)':<15}")
    print("-" * 80)

    # Find the most recent run for each scene
    scene_runs = {}
    for metrics_file in exp_dir.glob("*/metrics.json"):
        dir_name = metrics_file.parent.name
        # format is usually {scene_name}_{timestamp}
        parts = dir_name.split("_")
        if len(parts) >= 3:
            # The last two parts are date and time (e.g., 20260430_235838)
            scene_name = "_".join(parts[:-2])
        else:
            scene_name = "unknown"
            
        # keep only the latest one by string sort
        if scene_name not in scene_runs or dir_name > scene_runs[scene_name][0]:
            scene_runs[scene_name] = (dir_name, metrics_file)

    if not scene_runs:
        print("No metrics.json files found.")
        return

    averages = {'naive_hc': [], 'aware_hc': [], 'safety': [], 'plr': []}

    for scene, (d_name, m_file) in sorted(scene_runs.items()):
        with open(m_file) as f:
            data = json.load(f)
            
        nhc = data.get('naive_human_cost', 0)
        ahc = data.get('astar_human_cost', 0)
        saf = data.get('safety_improvement_pct', 0)
        plr = data.get('path_length_ratio', 1.0)
        
        averages['naive_hc'].append(nhc)
        averages['aware_hc'].append(ahc)
        averages['safety'].append(saf)
        averages['plr'].append(plr)

        print(f"{scene:<20} | {nhc:<10.4f} | {ahc:<10.4f} | {saf:>14.2f}% | {plr:>14.3f}x")

    print("-" * 80)
    n = len(scene_runs)
    avg_nhc = sum(averages['naive_hc']) / n
    avg_ahc = sum(averages['aware_hc']) / n
    avg_saf = sum(averages['safety']) / n
    avg_plr = sum(averages['plr']) / n
    
    print(f"{'AVERAGE':<20} | {avg_nhc:<10.4f} | {avg_ahc:<10.4f} | {avg_saf:>14.2f}% | {avg_plr:>14.3f}x")
    print("="*80 + "\n")
    
if __name__ == '__main__':
    main()
