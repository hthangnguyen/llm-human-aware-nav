#!/usr/bin/env python3
"""
Module 4 — Cost-Aware Path Planning (incorporating TASK-01).

Builds a grid graph over the navigable space.
Calculates A* paths avoiding high-probability human areas.
Compares against the naive shortest path.

Usage:
  python3 m4_plan_path.py \
    --scene-path /path/to/replica/room_0 \
    --output-dir /path/to/output \
    --horizon 30.0 \
    [--start x,y,z] [--goal x,y,z]
"""

import argparse
import json
import pathlib
import sys
import numpy as np
import habitat_sim
from scipy.spatial import KDTree
import heapq


GRID_SPACING = 0.25    # resolution of the grid graph (meters)
CONNECT_RADIUS = 0.40  # connect nodes within this distance
BETA = 100.0           # scaling factor for human proximity cost


def load_navmesh(scene_path: pathlib.Path) -> habitat_sim.NavMeshSettings:
    """Initialize PathFinder with the scene's navmesh."""
    navmesh_path = scene_path / 'habitat' / 'mesh_semantic.navmesh'
    if not navmesh_path.exists():
        raise FileNotFoundError(f"Navmesh missing: {navmesh_path}")

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(str(navmesh_path))
    if not pf.is_loaded:
        raise RuntimeError("Failed to load navmesh.")
    return pf


def build_navigable_graph(pf: habitat_sim.PathFinder, spacing: float, radius: float):
    """
    Sample a grid over the bounds, keep navigable points, build KDTree and adjacency list.
    """
    bounds = pf.get_bounds()
    min_b, max_b = bounds[0], bounds[1]
    
    # Generate grid
    x_range = np.arange(min_b[0], max_b[0], spacing)
    z_range = np.arange(min_b[2], max_b[2], spacing)
    
    # We sample at a fixed height from the bounds, but snap to navmesh
    y = min_b[1] + 0.1 

    nodes = []
    for x in x_range:
        for z in z_range:
            pt = np.array([x, y, z], dtype=np.float32)
            # snap to navmesh
            pt_snapped = pf.snap_point(pt)
            if not np.isnan(pt_snapped[0]):
                pt_snapped_np = np.array(pt_snapped)
                # avoid duplicates
                if not nodes or np.min(np.linalg.norm(np.array(nodes) - pt_snapped_np, axis=1)) > spacing * 0.5:
                    nodes.append(pt_snapped_np)

    nodes = np.array(nodes)
    if len(nodes) == 0:
        raise ValueError("Failed to sample navigable nodes.")

    print(f"Sampled {len(nodes)} navigable nodes.")
    
    tree = KDTree(nodes)
    edges = {}
    for i in range(len(nodes)):
        neighbors = tree.query_ball_point(nodes[i], radius)
        edges[i] = [n for n in neighbors if n != i]
        
    return nodes, edges, tree


def compute_human_costs(nodes: np.ndarray, traj_data: dict) -> np.ndarray:
    """
    Compute human proximity cost for each node in the graph.
    Cost = sum_j (prob_j / (dist(node, obj_j)^2 + epsilon))
    """
    probs = np.array(traj_data['probs'])
    obj_positions = np.array(traj_data['positions'])
    
    costs = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        # Distances from this node to all objects
        dists = np.linalg.norm(obj_positions - node, axis=1)
        # Add epsilon to prevent division by zero
        sq_dists = dists**2 + 0.1
        costs[i] = np.sum(probs / sq_dists)
        
    return costs


def astar(start_idx: int, goal_idx: int, nodes: np.ndarray, edges: dict, human_costs: np.ndarray, beta: float):
    """Run A* search on the graph."""
    open_set = []
    heapq.heappush(open_set, (0.0, start_idx))
    
    came_from = {}
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[start_idx] = 0.0
    
    f_score = {i: float('inf') for i in range(len(nodes))}
    f_score[start_idx] = np.linalg.norm(nodes[start_idx] - nodes[goal_idx])
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [nodes[i].tolist() for i in path]
            
        for neighbor in edges[current]:
            dist = np.linalg.norm(nodes[current] - nodes[neighbor])
            # Cost = distance * (1 + beta * human_proximity_at_neighbor)
            edge_cost = dist * (1.0 + beta * human_costs[neighbor])
            
            tentative_g = g_score[current] + edge_cost
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = np.linalg.norm(nodes[neighbor] - nodes[goal_idx])
                f_score[neighbor] = tentative_g + h
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
    return None # No path found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-path', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--horizon',    type=float, default=30.0)
    ap.add_argument('--start',      help='comma-separated x,y,z')
    ap.add_argument('--goal',       help='comma-separated x,y,z')
    ap.add_argument('--no-clio',    action='store_true', help='Disable Clio map filtering')
    args = ap.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    traj_path  = output_dir / 'trajectories.json'
    
    if not traj_path.exists():
        print(f'[ERROR] {traj_path} missing. Run Module 2 first.')
        sys.exit(1)

    with open(traj_path) as f:
        trajectories = json.load(f)
        
    horizon_key = str(args.horizon)
    if horizon_key not in trajectories:
        print(f'[WARN] Horizon {horizon_key} not found. Using first available.')
        horizon_key = list(trajectories.keys())[0]
        
    traj_data = trajectories[horizon_key]
    
    if args.no_clio:
        sg_path = output_dir / 'scene_graph.json'
    else:
        sg_path = output_dir / 'scene_graph_clio.json'
        
    with open(sg_path) as f:
        valid_sg = json.load(f)
        
    valid_node_ids = set(valid_sg.keys())
    
    filtered_probs = []
    filtered_positions = []
    
    for prob, nid, pos in zip(traj_data['probs'], traj_data.get('node_ids', []), traj_data['positions']):
        if nid in valid_node_ids:
            filtered_probs.append(prob)
            filtered_positions.append(pos)
            
    traj_data['probs'] = filtered_probs
    traj_data['positions'] = filtered_positions

    print("Loading NavMesh...")
    pf = load_navmesh(pathlib.Path(args.scene_path))
    
    print("Building navigable grid graph...")
    nodes, edges, tree = build_navigable_graph(pf, GRID_SPACING, CONNECT_RADIUS)
    
    print("Computing human proximity costs...")
    human_costs = compute_human_costs(nodes, traj_data)

    # Determine start and goal
    if args.start and args.goal:
        start_pt = np.array([float(v) for v in args.start.split(',')])
        goal_pt  = np.array([float(v) for v in args.goal.split(',')])
    else:
        # Auto: furthest navigable pair
        start_pt = pf.get_random_navigable_point()
        best_dist = 0.0
        goal_pt = start_pt
        for _ in range(100):
            cand = pf.get_random_navigable_point()
            d = np.linalg.norm(cand - start_pt)
            if d > best_dist:
                best_dist = d
                goal_pt = cand
                
    print(f"Start: {np.array(start_pt).tolist()}")
    print(f"Goal:  {np.array(goal_pt).tolist()}")
    
    # Snap to nearest graph nodes
    _, start_idx = tree.query(start_pt)
    _, goal_idx  = tree.query(goal_pt)
    
    print("Computing naive geodesic shortest path (habitat-sim baseline)...")
    path_finder_path = habitat_sim.ShortestPath()
    path_finder_path.requested_start = start_pt
    path_finder_path.requested_end = goal_pt
    pf.find_path(path_finder_path)
    naive_path = [np.array(pt).tolist() for pt in path_finder_path.points]

    print(f"Computing Cost-Aware A* path (beta={BETA})...")
    astar_path = astar(start_idx, goal_idx, nodes, edges, human_costs, BETA)
    
    if not astar_path:
        print("[ERROR] A* could not find a path.")
        astar_path = []

    # Calculate metrics
    def path_length(pts):
        return sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1)) if len(pts)>1 else 0.0
        
    def avg_human_cost(pts):
        if len(pts) == 0: return 0.0
        c = 0.0
        for pt in pts:
            _, idx = tree.query(pt)
            c += human_costs[idx]
        return c / len(pts)

    naive_len = path_length(naive_path)
    astar_len = path_length(astar_path)
    naive_hc  = avg_human_cost(naive_path)
    astar_hc  = avg_human_cost(astar_path)
    
    print("\n--- RESULTS ---")
    print(f"Naive Path: Length={naive_len:.2f}m, Avg Human Cost={naive_hc:.4f}")
    print(f"A*    Path: Length={astar_len:.2f}m, Avg Human Cost={astar_hc:.4f}")
    
    result = {
        'start': np.array(start_pt).tolist(),
        'goal': np.array(goal_pt).tolist(),
        'horizon': args.horizon,
        'naive_path': naive_path,
        'astar_path': astar_path,
        'metrics': {
            'naive_length': naive_len,
            'astar_length': astar_len,
            'naive_human_cost': naive_hc,
            'astar_human_cost': astar_hc
        }
    }
    
    out_path = output_dir / 'path.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved path.json → {out_path}")

if __name__ == '__main__':
    main()
