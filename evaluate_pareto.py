#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Pareto Checkpoint Evaluator — 筛选 forward 满分下 pose/energy/time 最优的模型

Usage:
    python evaluate_pareto.py scores.jsonl --forward-threshold 90 --top-k 10
    python evaluate_pareto.py scores.csv --format csv --forward-threshold 90

Output:
    - 命令行表格：Top-K Pareto 最优 episode/checkpoint
    - pareto_report.json：详细数据
    - pareto_front.png：可视化（需要 matplotlib）
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip malformed line {i}: {e}")
    return episodes


def parse_csv(path: str) -> List[Dict[str, Any]]:
    import csv
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            episodes.append(row)
    return episodes


def compute_composite(e: Dict[str, Any], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, w in weights.items():
        score += w * e.get(key, 0.0)
    return score


def pareto_filter(
    episodes: List[Dict[str, Any]],
    forward_threshold: float = 90.0,
    forward_key: str = "forward_score",
    pose_key: str = "pose_score",
    energy_key: str = "energy_score",
    time_key: str = "time_score",
    composite_weights: Dict[str, float] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if composite_weights is None:
        composite_weights = {pose_key: 1.0, energy_key: 1.0, time_key: 1.0}

    filtered = []
    for e in episodes:
        fwd = e.get(forward_key, 0.0)
        if fwd >= forward_threshold:
            e["_composite"] = compute_composite(e, composite_weights)
            e["_forward"] = fwd
            filtered.append(e)

    filtered.sort(key=lambda x: x["_composite"], reverse=True)
    return filtered[:top_k]


def compute_pareto_front(
    episodes: List[Dict[str, Any]],
    forward_key: str = "forward_score",
    pose_key: str = "pose_score",
    energy_key: str = "energy_score",
    time_key: str = "time_score",
) -> List[Dict[str, Any]]:
    candidates = [e for e in episodes if e.get(forward_key, 0) > 50]
    front = []
    for e in candidates:
        p, en, t = e.get(pose_key, 0), e.get(energy_key, 0), e.get(time_key, 0)
        dominated = False
        for other in candidates:
            if other is e:
                continue
            op, oen, ot = other.get(pose_key, 0), other.get(energy_key, 0), other.get(time_key, 0)
            if op >= p and oen >= en and ot >= t and (op > p or oen > en or ot > t):
                dominated = True
                break
        if not dominated:
            e["_composite"] = p + en + t
            e["_forward"] = e.get(forward_key, 0)
            front.append(e)
    front.sort(key=lambda x: x["_composite"], reverse=True)
    return front


def print_table(results: List[Dict[str, Any]], title: str = "Pareto Top Results"):
    if not results:
        print("No episodes meet the forward threshold.")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Rank':<6}{'Forward':<10}{'Pose':<10}{'Energy':<10}{'Time':<10}{'Composite':<12}{'Checkpoint':<25}"
    print(header)
    print("-" * 80)
    for i, r in enumerate(results, 1):
        ckpt = r.get("checkpoint", r.get("episode_id", r.get("id", "N/A")))
        row = f"{i:<6}{r['_forward']:<10.1f}{r.get('pose_score', 0):<10.1f}"
        row += f"{r.get('energy_score', 0):<10.1f}{r.get('time_score', 0):<10.1f}"
        row += f"{r['_composite']:<12.1f}{str(ckpt):<25}"
        print(row)
    print("=" * 80)


def plot_pareto(episodes: List[Dict[str, Any]], output_path: str = "pareto_front.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fwd = [e.get("forward_score", 0) for e in episodes]
    pose = [e.get("pose_score", 0) for e in episodes]
    energy = [e.get("energy_score", 0) for e in episodes]
    time = [e.get("time_score", 0) for e in episodes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(fwd, pose, alpha=0.3, s=10, c="blue")
    axes[0].set_xlabel("Forward Score")
    axes[0].set_ylabel("Pose Score")
    axes[0].set_title("Pose vs Forward")
    axes[0].axvline(x=90, color="red", linestyle="--", label="threshold=90")
    axes[0].legend()

    axes[1].scatter(fwd, energy, alpha=0.3, s=10, c="green")
    axes[1].set_xlabel("Forward Score")
    axes[1].set_ylabel("Energy Score")
    axes[1].set_title("Energy vs Forward")
    axes[1].axvline(x=90, color="red", linestyle="--", label="threshold=90")
    axes[1].legend()

    axes[2].scatter(fwd, time, alpha=0.3, s=10, c="orange")
    axes[2].set_xlabel("Forward Score")
    axes[2].set_ylabel("Time Score")
    axes[2].set_title("Time vs Forward")
    axes[2].axvline(x=90, color="red", linestyle="--", label="threshold=90")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Pareto plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pareto checkpoint evaluator")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--format", choices=["jsonl", "csv", "auto"], default="auto")
    parser.add_argument("--forward-threshold", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="pareto_report.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--pareto-front", action="store_true")
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--time-weight", type=float, default=1.0)
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = "csv" if args.input.endswith(".csv") else "jsonl"

    print(f"Loading data from: {args.input} (format: {fmt})")
    episodes = parse_csv(args.input) if fmt == "csv" else parse_jsonl(args.input)
    print(f"Total episodes loaded: {len(episodes)}")

    if not episodes:
        print("No data found. Exiting.")
        sys.exit(1)

    composite_weights = {
        "pose_score": args.pose_weight,
        "energy_score": args.energy_weight,
        "time_score": args.time_weight,
    }

    if args.pareto_front:
        print("\nComputing true Pareto frontier...")
        results = compute_pareto_front(episodes)
        print_table(results, title="True Pareto Frontier")
    else:
        print(f"\nFiltering forward >= {args.forward_threshold}...")
        results = pareto_filter(episodes, args.forward_threshold, composite_weights=composite_weights, top_k=args.top_k)
        print_table(results, title=f"Top-{args.top_k} Pareto Optimal")

    report = {
        "config": {"forward_threshold": args.forward_threshold, "top_k": args.top_k, "composite_weights": composite_weights},
        "total_episodes": len(episodes),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {args.output}")

    if args.plot:
        plot_pareto(episodes)

    if episodes:
        fwd_scores = [e.get("forward_score", 0) for e in episodes]
        pose_scores = [e.get("pose_score", 0) for e in episodes]
        energy_scores = [e.get("energy_score", 0) for e in episodes]
        time_scores = [e.get("time_score", 0) for e in episodes]
        print(f"\n{'='*50}")
        print("  Dataset Statistics")
        print(f"{'='*50}")
        print(f"Forward:  mean={sum(fwd_scores)/len(fwd_scores):.1f}, max={max(fwd_scores):.1f}")
        print(f"Pose:     mean={sum(pose_scores)/len(pose_scores):.1f}, max={max(pose_scores):.1f}")
        print(f"Energy:   mean={sum(energy_scores)/len(energy_scores):.1f}, max={max(energy_scores):.1f}")
        print(f"Time:     mean={sum(time_scores)/len(time_scores):.1f}, max={max(time_scores):.1f}")


if __name__ == "__main__":
    main()
