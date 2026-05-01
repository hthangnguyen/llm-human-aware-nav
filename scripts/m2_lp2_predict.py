#!/usr/bin/env python3
"""
Module 2 — LP2 Human Trajectory Prediction.

ISP: Ollama LLM predicts which objects a person visits next and for how long.
PTP: CTMC propagates spatial probability over scene graph nodes.

Includes fixes from TASK-05 (ISP Logging) and TASK-06 (State-dependent CTMC).

Usage:
  python3 m2_lp2_predict.py \
    --output-dir /path/to/output \
    --model llama3.2:3b \
    --compat-matrix ../data/compat_matrix.json \
    [--start-class chair]
"""

import argparse
import json
import os
import pathlib
import sys
import time
import requests
import numpy as np
from scipy.linalg import expm
from collections import Counter


MOCK_LLM          = os.environ.get('MOCK_LLM', '0') == '1'
OLLAMA_URL        = 'http://localhost:11434/api/generate'
REQUEST_TIMEOUT_S = 90
WALK_SPEED_MS     = 1.2   # meters/second (average indoor walking speed)


# ---------- Ollama helper ----------

def ollama_call(prompt: str, model: str) -> str:
    """Call Ollama. Returns raw response string. Raises on failure."""
    resp = requests.post(
        OLLAMA_URL,
        json={'model': model, 'prompt': prompt, 'stream': False},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json().get('response', '').strip()


# ---------- ISP: Interaction Sequence Predictor ----------

def isp_predict(scene_graph: dict, start_class: str,
                model: str, output_dir: pathlib.Path) -> tuple[dict[str, float], dict[str, float]]:
    """
    Ask Ollama which objects a person visits next after interacting with start_class.

    Returns:
      probs:     dict[class_name -> probability float]. Sum <= 1.0.
      durations: dict[class_name -> expected seconds at that object].
    """
    all_classes = sorted(set(v['class'] for v in scene_graph.values()))
    n           = len(all_classes)

    uniform_probs     = {c: 1.0 / n for c in all_classes}
    uniform_durations = {c: 30.0    for c in all_classes}

    log_entry = {
        "timestamp": time.time(),
        "start_class": start_class,
        "raw_response": "",
        "parsed_ok": False,
        "probs_sum": 1.0,
        "used_fallback": True
    }

    if MOCK_LLM:
        print('[MOCK] ISP: returning uniform distribution')
        return uniform_probs, uniform_durations

    prompt = (
        f'You are modeling human behavior in an indoor room.\n'
        f'Objects in the room: {", ".join(all_classes)}.\n'
        f'A person just finished using: {start_class}.\n\n'
        f'Predict the next objects they will interact with.\n'
        f'Rules:\n'
        f'- List at most 6 objects.\n'
        f'- Each probability must be between 0 and 1.\n'
        f'- All probabilities must sum to 1.0 or less.\n'
        f'- duration_s is expected seconds spent at that object.\n'
        f'- Use only class names from the list above.\n\n'
        f'Respond with ONLY a JSON array. No explanation. No markdown. Example:\n'
        f'[{{"object":"chair","prob":0.4,"duration_s":120}},'
        f'{{"object":"table","prob":0.3,"duration_s":60}}]'
    )

    try:
        raw = ollama_call(prompt, model)
        log_entry["raw_response"] = raw

        # Extract JSON array robustly
        start_idx = raw.find('[')
        end_idx   = raw.rfind(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f'No JSON array found in response: {raw[:200]}')
        predictions = json.loads(raw[start_idx:end_idx + 1])

        # Validate and extract
        probs     = {}
        durations = {}
        for p in predictions:
            name = str(p.get('object', '')).strip()
            prob = float(p.get('prob', 0.0))
            dur  = max(float(p.get('duration_s', 30.0)), 1.0)
            if name in all_classes and 0 < prob <= 1.0:
                probs[name]     = prob
                durations[name] = dur

        if not probs:
            raise ValueError('No valid predictions extracted from LLM response.')

        # Normalize if sum > 1.0
        total = sum(probs.values())
        if total > 1.0:
            probs = {k: v / total for k, v in probs.items()}
            total = 1.0

        print(f'ISP predictions: {probs}')
        
        log_entry["parsed_ok"] = True
        log_entry["probs_sum"] = total
        log_entry["used_fallback"] = False

        # Write log
        with open(output_dir / 'isp_log.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return probs, durations

    except Exception as e:
        print(f'[WARN] ISP LLM failed: {e}. Falling back to uniform distribution.')
        # Write log
        with open(output_dir / 'isp_log.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        return uniform_probs, uniform_durations


# ---------- PTP: CTMC builder (TASK-06 Updated) ----------

def build_ctmc(scene_graph: dict,
               isp_probs: dict[str, float],
               isp_durations: dict[str, float],
               compat_matrix: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build CTMC generator matrix Q over scene graph nodes.
    Includes TASK-06 fixes:
      - State-dependent compatibility (compat_matrix).
      - Sojourn time encoding (using isp_durations).
    """
    node_ids  = list(scene_graph.keys())
    positions = np.array([scene_graph[n]['center'] for n in node_ids], dtype=np.float64)
    N         = len(node_ids)
    Q         = np.zeros((N, N), dtype=np.float64)

    default_compat = compat_matrix.get("default", 0.3)

    for i in range(N):
        class_i = scene_graph[node_ids[i]]['class']
        
        # Calculate raw unnormalized off-diagonal transition rates
        row_rates = np.zeros(N)
        for j in range(N):
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 1e-3:
                continue   # avoid division by zero
            
            class_j = scene_graph[node_ids[j]]['class']
            
            # ISP probability for class_j
            p_act_j = isp_probs.get(class_j, 0.005)
            
            # Compatibility score between class_i and class_j
            compat = compat_matrix.get(class_i, {}).get(class_j, default_compat)
            
            # Base transition rate (inversely proportional to dist)
            row_rates[j] = (WALK_SPEED_MS / dist) * p_act_j * compat

        sum_off_diag = row_rates.sum()
        
        # Apply sojourn time (expected duration at node i)
        # Expected duration = -1 / Q[i,i]. So Q[i,i] = -1 / duration
        dur_i = isp_durations.get(class_i, 30.0)
        holding_rate = 1.0 / dur_i
        
        if sum_off_diag > 1e-9:
            # Scale off-diagonal rates so they sum to holding_rate
            row_rates = row_rates * (holding_rate / sum_off_diag)
            Q[i, :] = row_rates
            Q[i, i] = -holding_rate
        else:
            Q[i, i] = 0.0

    return Q, node_ids, positions


def propagate_ctmc(Q: np.ndarray, p0: np.ndarray,
                   horizons: list[float]) -> dict[float, np.ndarray]:
    """
    Solve Kolmogorov forward equation: p(t) = expm(Q*t) @ p0
    """
    results  = {}
    n        = len(p0)
    uniform  = np.ones(n) / n

    for t in horizons:
        Qt = Q * t

        max_eig = float(np.max(np.real(np.linalg.eigvals(Qt))))
        if max_eig > 500:
            print(f'[WARN] t={t}s: max eigenvalue {max_eig:.1f}. '
                  'CTMC may be ill-conditioned.')

        Pt = expm(Qt)

        if np.isnan(Pt).any() or np.isinf(Pt).any():
            print(f'[WARN] t={t}s: expm produced NaN/inf. Falling back.')
            results[t] = uniform.copy()
            continue

        pt = np.real(Pt @ p0)
        pt = np.clip(pt, 0, None)
        total = pt.sum()
        if total > 1e-9:
            pt /= total
        else:
            print(f'[WARN] t={t}s: probability sum near zero. Using uniform.')
            pt = uniform.copy()

        results[t] = pt

    return results


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir',   required=True)
    ap.add_argument('--model',        default='llama3.2:3b')
    ap.add_argument('--compat-matrix', required=True)
    ap.add_argument('--start-class',  default=None)
    ap.add_argument('--horizons',     nargs='+', type=float,
                    default=[10.0, 30.0, 60.0])
    args = ap.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    sg_path    = output_dir / 'scene_graph.json'
    compat_path= pathlib.Path(args.compat_matrix)

    if not sg_path.exists():
        print(f'[ERROR] {sg_path} not found. Run Module 1 first.')
        sys.exit(1)

    sg = json.load(open(sg_path))
    if not sg:
        print('[ERROR] Scene graph is empty.')
        sys.exit(1)

    if not compat_path.exists():
        print(f'[ERROR] Compat matrix {compat_path} not found.')
        sys.exit(1)
        
    compat_matrix = json.load(open(compat_path))

    # ---------- Select start class ----------
    all_classes = [v['class'] for v in sg.values()]
    if args.start_class:
        if args.start_class not in set(all_classes):
            print(f'[ERROR] --start-class "{args.start_class}" not in scene graph.')
            sys.exit(1)
        start_class = args.start_class
    else:
        # Auto: most frequent valid class
        invalid_classes = {'unknown', 'undefined', 'None', 'null'}
        counts = Counter(c for c in all_classes if c not in invalid_classes)
        if not counts:
            print('[ERROR] All objects are invalid classes. Cannot run LP2.')
            sys.exit(1)
        start_class = counts.most_common(1)[0][0]
        print(f'Auto-selected start_class: "{start_class}" (most frequent)')

    # ---------- ISP ----------
    isp_probs, isp_durations = isp_predict(sg, start_class, args.model, output_dir)

    # ---------- CTMC ----------
    Q, node_ids, positions = build_ctmc(sg, isp_probs, isp_durations, compat_matrix)

    # Initial distribution: uniform over all nodes of start_class
    start_indices = [i for i, nid in enumerate(node_ids)
                     if sg[nid]['class'] == start_class]
    if not start_indices:
        start_indices = [0]
    p0 = np.zeros(len(node_ids))
    for idx in start_indices:
        p0[idx] = 1.0 / len(start_indices)

    distributions = propagate_ctmc(Q, p0, args.horizons)

    # ---------- Save ----------
    result = {}
    for t, pt in distributions.items():
        top3_raw = sorted(
            zip(pt.tolist(), node_ids),
            key=lambda x: x[0],
            reverse=True
        )[:3]
        top3 = [
            {'prob': round(p, 4), 'node_id': nid, 'class': sg[nid]['class']}
            for p, nid in top3_raw
        ]
        print(f't={t}s top-3: {[(e["class"], round(e["prob"],3)) for e in top3]}')

        result[str(float(t))] = {
            'probs':     pt.tolist(),
            'node_ids':  node_ids,
            'positions': positions.tolist(),
            'top3':      top3,
        }

    out_path = output_dir / 'trajectories.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved trajectories.json → {out_path}')


if __name__ == '__main__':
    main()
