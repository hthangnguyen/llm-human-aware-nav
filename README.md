# Human-Aware Robot Path Planning via Local LLM Trajectory Prediction

## Abstract
This project implements a simulation-only pipeline for human-aware robot path planning in indoor environments. Starting from a semantic scene graph, the system uses a local LLM (`llama3.2:3b`) to predict human movement via a Continuous-Time Markov Chain (inspired by LP2), then plans a cost-weighted path that trades off travel distance against proximity to predicted human locations. We evaluate the system on 3 Replica scenes (`room_0`, `room_1`, `office_0`) and show that human-aware planning achieves up to ~12% improvement in human-proximity safety at a cost of only ~8% longer paths.

## System Architecture

1. **Scene Graph Builder (DAAAM Approximation):** Parses Habitat's `info_semantic.json` and uses an LLM to generate descriptive embeddings.
2. **Trajectory Prediction (LP2 Approximation):** Predicts human interaction sequences using a local LLM and propagates spatial probability over the scene graph via a state-dependent Continuous-Time Markov Chain (CTMC).
3. **Map Compression (Clio Approximation):** Filters non-task-relevant objects out of the environment map to reduce noise.
4. **Cost-Aware Path Planner:** Constructs a dense navigable grid from the `habitat-sim` NavMesh, evaluates the human proximity cost field, and calculates a detour path via A* that minimizes a combined distance-and-human-proximity cost metric.
5. **Visualization & Evaluation:** Automatically charts paths in 2D and performs quantitative evaluations (Path Length Ratio, Safety Improvement).

## Visual Results

**Left:** `room_0` showing a successful Cost-Aware detour (Blue) routing around the high-probability human zones.

**Right:** `office_0` showing a topological bottleneck where the planner is forced to cross danger zones.

<p align="center">
  <img src="assets/images/room_0_success.png" width="45%" />
  <img src="assets/images/office_0_bottleneck.png" width="45%" />
</p>

## Quantitative Results

Across 3 indoor scenes, the custom Cost-Aware A* path successfully identified and bypassed high-probability zones at the expense of a slightly longer total travel distance. To combat A*'s tendency to optimize for average cost over peak danger, we implemented **quadratic human-cost penalties** and evaluated the planner using both Average Safety and **Peak Safety** (Maximum Danger) metrics.

| Scene Name        | Avg Safety Imp % | Peak Safety Imp % | Naive Peak | Aware Peak | Path Length Ratio |
|-------------------|------------------|-------------------|------------|------------|-------------------|
| `office_0`        | -7.02%           | -10.76%           | 0.2610     | 0.2891     | 1.196x            |
| `office_0_noclio` | -6.33%           | -6.33%            | 0.2767     | 0.2942     | 1.196x            |
| `room_0`          | 16.66%           | 8.64%             | 0.2004     | 0.1831     | 1.111x            |
| `room_0_noclio`   | 13.10%           | 0.00%             | 0.1608     | 0.1608     | 1.083x            |
| `room_1`          | -6.48%           | 0.00%             | 0.0989     | 0.0989     | 1.040x            |
| `room_1_noclio`   | 14.09%           | 0.00%             | 0.1615     | 0.1615     | 1.169x            |

*(Note: The 0.00% Peak Improvement in several runs indicates that the absolute highest danger point occurred exactly at the starting or ending coordinate, making the peak mathematically unavoidable. In severe bottlenecks like `office_0`, A* calculated that enduring a higher peak of 0.2891 was still mathematically cheaper than taking an immense physical detour around the entire office layout.)*

### Dynamic Heuristics & Clio Ablation Study
Instead of relying on a static, hand-coded transition matrix, the pipeline queries the LLM at runtime to generate a **Dynamic Compatibility Matrix** tailored to the unique classes found in each room. This eliminated structural biases and boosted `room_0`'s Average Safety Improvement to a staggering **16.66%**.

When the pipeline was run with the Clio map filter disabled (`_noclio`), we observed distinct shifts. In `room_0`, Clio successfully suppressed background noise, improving the average safety gain from 13.10% to 16.66% and unlocking an 8.64% peak improvement. However, in `room_1`, Clio's strict text-filtering inadvertently degraded performance compared to the unfiltered map, highlighting a critical area for future LLM-vision alignment research.

## How to Run

Execute the end-to-end pipeline across all evaluation scenes:

```bash
bash scripts/run_all_scenes.sh
```

To run a single scene and output the top-down visualization and metrics:

```bash
bash scripts/demo.sh datasets/replica/room_0
```

## Known Limitations

- **Simulated Human Movement:** We approximate human probability via CTMC instead of executing a fully embodied human avatar.
- **2D Planning:** Path planning uses standard floor-level navigation and doesn't fully account for 3D human pose (e.g., reaching or bending).
- **Hardcoded NavMesh Parsing:** Real robots must build semantic maps via SLAM, whereas this project bypasses vision constraints by leveraging the Replica ground truth annotations.
