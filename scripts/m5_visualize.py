#!/usr/bin/env python3
"""
Module 5 — Visualization.

Generates a 2D top-down plot (X,Z plane) of the scene graph objects
and the planned paths (Naive vs A*).
Saves to topdown_paths.png.

Usage:
  python3 m5_visualize.py \
    --output-dir /path/to/output
"""

import argparse
import json
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use('Agg') # Headless mode
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("[ERROR] matplotlib is required for Module 5. Please install it.")
    sys.exit(1)


def plot_scene(output_dir: pathlib.Path):
    sg_path   = output_dir / 'scene_graph.json'
    path_path = output_dir / 'path.json'

    if not sg_path.exists() or not path_path.exists():
        print(f"[ERROR] Required files missing in {output_dir}")
        sys.exit(1)

    with open(sg_path) as f:
        scene_graph = json.load(f)
    with open(path_path) as f:
        path_data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Plot objects as rectangles
    for oid, obj in scene_graph.items():
        cx, cy, cz = obj['center']
        sx, sy, sz = obj['size']
        
        # In Habitat, Y is usually up. We plot X and Z.
        # Bottom-left corner for the rectangle
        rect_x = cx - sx / 2
        rect_z = cz - sz / 2
        
        rect = patches.Rectangle(
            (rect_x, rect_z), sx, sz,
            linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5
        )
        ax.add_patch(rect)
        
        # Add a subtle label for large objects or specific classes
        if sx * sz > 0.5 or obj['class'] in ['bed', 'sofa', 'table']:
            ax.text(cx, cz, obj['class'], fontsize=6, ha='center', va='center', color='black')

    # 2. Plot Naive Path (Red)
    naive_pts = path_data.get('naive_path', [])
    if naive_pts:
        nx = [p[0] for p in naive_pts]
        nz = [p[2] for p in naive_pts]
        ax.plot(nx, nz, 'r--', linewidth=2, label='Naive Shortest Path', marker='o', markersize=3)

    # 3. Plot A* Path (Blue)
    astar_pts = path_data.get('astar_path', [])
    if astar_pts:
        ax_x = [p[0] for p in astar_pts]
        ax_z = [p[2] for p in astar_pts]
        ax.plot(ax_x, ax_z, 'b-', linewidth=2, label='Cost-Aware A* Path', marker='o', markersize=3)

    # 4. Mark Start and Goal
    start = path_data['start']
    goal = path_data['goal']
    ax.plot(start[0], start[2], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[2], 'r*', markersize=12, label='Goal')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title('Top-Down View: Path Planning vs Human Proximity')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Invert Z axis if needed to match standard top-down conventions
    # (Optional, depending on Habitat's exact coordinate frame)
    # ax.invert_yaxis()

    out_file = output_dir / 'topdown_paths.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization → {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    plot_scene(output_dir)

if __name__ == '__main__':
    main()
