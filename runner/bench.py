"""leanBench orchestrator.

Usage:
    python runner/bench.py --label my-laptop

Runs each workload in a separate subprocess of the Rust runner binary,
samples CPU/memory during the run, then writes one JSON file per run under
`results/<timestamp>__<fingerprint>.json`.

The output is committed as-is (no post-processing, no index rebuild — that
happens at deploy time in GH Actions).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from runner import sysinfo
from runner.sampler import ResourceSampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

@dataclass(frozen=True)
class Workload:
    # argv passed to the Rust binary; aggregates take positional `n` (flat),
    # `fan n` (tree), or `root_fan mid_fan n` (deep).
    cli_args: list[str]
    # workload identifier emitted in the JSON record (`workload` field).
    name: str
    in_default_set: bool = field(kw_only=True)
    samples_override: int | None = field(default=None, kw_only=True)


# samples_override on the heaviest tree workloads: cv ≤ 0.025 on these
# four, so n=5 gives a 95% CI on the mean of ≤ ±2.2% — tight enough for
# cross-machine comparison and topology fitting. Keeps the full matrix
# under a ~30-min wall budget on 8-core remote VMs.
ALL_WORKLOADS: list[Workload] = [
    Workload(["aggregate-flat", "125"],     "aggregate.flat_125_r2",   in_default_set=True),
    Workload(["aggregate-flat", "250"],     "aggregate.flat_250_r2",   in_default_set=True),
    Workload(["aggregate-flat", "500"],     "aggregate.flat_500_r2",   in_default_set=True),
    Workload(["aggregate-flat", "1000"],    "aggregate.flat_1000_r2",  in_default_set=True),
    Workload(["aggregate-tree", "2", "125"], "aggregate.tree_2x125_r2", in_default_set=True),
    Workload(["aggregate-tree", "2", "250"], "aggregate.tree_2x250_r2", in_default_set=True),
    Workload(["aggregate-tree", "2", "500"], "aggregate.tree_2x500_r2", in_default_set=True),
    Workload(["aggregate-tree", "4", "125"], "aggregate.tree_4x125_r2", in_default_set=True),
    Workload(["aggregate-tree", "4", "250"], "aggregate.tree_4x250_r2", in_default_set=True),
    Workload(["aggregate-tree", "4", "500"], "aggregate.tree_4x500_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-tree", "8", "125"], "aggregate.tree_8x125_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-tree", "8", "250"], "aggregate.tree_8x250_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-tree", "8", "500"], "aggregate.tree_8x500_r2", in_default_set=True, samples_override=5),
    # Deep (3-level) recursion: root_fan x mid_fan x leaf_n. Probes aggregate-fast-
    # aggregate-often vs the wide 2-level tree above. samples_override scales down
    # with cost — the heaviest (4x5x500 = 10k sigs) runs at ~2 min/sample even on
    # an 8-core VM, so it gets n=3; the 4x4x* pair and 2x2x500 get n=5.
    Workload(["aggregate-deep", "2", "2", "125"], "aggregate.deep_2x2x125_r2", in_default_set=True),
    Workload(["aggregate-deep", "2", "2", "250"], "aggregate.deep_2x2x250_r2", in_default_set=True),
    Workload(["aggregate-deep", "2", "2", "500"], "aggregate.deep_2x2x500_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-deep", "4", "4", "125"], "aggregate.deep_4x4x125_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-deep", "4", "4", "250"], "aggregate.deep_4x4x250_r2", in_default_set=True, samples_override=5),
    Workload(["aggregate-deep", "4", "5", "500"], "aggregate.deep_4x5x500_r2", in_default_set=True, samples_override=3),
    # 2x4 vs 4x2 (both 1000 sigs): does fanning wide at the root or at the mid tier win?
    Workload(["aggregate-deep", "2", "4", "125"], "aggregate.deep_2x4x125_r2", in_default_set=True),
    Workload(["aggregate-deep", "4", "2", "125"], "aggregate.deep_4x2x125_r2", in_default_set=True),
    # split / merge_many strategies on parent type-2 (K x N at LOG_INV_RATE_PROD=2).
    # Heavier than flat/tree because every iteration rebuilds the parent type-2;
    # samples_override=5 on the merge / strategy variants keeps the budget bounded.
    Workload(["split", "125", "2"],                "aggregate.split_2x125_r2",                in_default_set=False),
    Workload(["split", "250", "2"],                "aggregate.split_2x250_r2",                in_default_set=False),
    Workload(["split", "500", "2"],                "aggregate.split_2x500_r2",                in_default_set=False, samples_override=5),
    Workload(["merge-split-and-original", "125", "2"], "aggregate.merge_split_and_original_2x125_r2", in_default_set=False),
    Workload(["merge-split-and-original", "250", "2"], "aggregate.merge_split_and_original_2x250_r2", in_default_set=False, samples_override=5),
    Workload(["merge-split-and-original", "500", "2"], "aggregate.merge_split_and_original_2x500_r2", in_default_set=False, samples_override=5),
    Workload(["merge-split-and-leaves", "125", "2", "125"], "aggregate.merge_split_and_leaves_2x125x125_r2", in_default_set=False),
    Workload(["merge-split-and-leaves", "250", "2", "250"], "aggregate.merge_split_and_leaves_2x250x250_r2", in_default_set=False, samples_override=5),
    Workload(["merge-split-and-leaves", "500", "2", "500"], "aggregate.merge_split_and_leaves_2x500x500_r2", in_default_set=False, samples_override=5),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None,
                    help="Human-readable nickname for this machine. "
                         "Defaults to hostname (or CPU-slug if hostname is generic).")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--samples", type=int, default=10,
                    help="Timed samples per workload (warm-up excluded)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", action="append", default=None,
                    help="Run only these workload names; repeatable.")
    ap.add_argument("--notes", default="",
                    help="Free-form note attached to the run record")
    ap.add_argument("--features", default=None,
                    help="cargo feature(s) for the runner build (comma-separated). "
                         "Pick one API mode: 'api-leansig' (devnet4 / devnet5 leanVM) or "
                         "'api-xmss' (main leanVM). Default uses Cargo.toml's default feature.")
    return ap.parse_args()


def select_workloads(args) -> list[Workload]:
    if args.only:
        requested = set(args.only)
        selected = [w for w in ALL_WORKLOADS if w.name in requested]
        missing = requested - {w.name for w in selected}
        if missing:
            sys.exit(f"unknown workload(s): {', '.join(sorted(missing))}")
        return selected
    return [w for w in ALL_WORKLOADS if w.in_default_set]


RUST_DIR = ROOT / "workloads"


def build_runner(features: str | None = None) -> Path:
    """Build the Rust workload binary in release mode.

    Sets `RUSTFLAGS=-C target-cpu=native` in the subprocess env so deps
    (specifically leanVM's KoalaBear backend) compile with the host's
    SIMD target features. Without this, the build defaults to baseline
    x86-64 features, falls into leanVM's no-SIMD path
    (`type Packing = Self`), and trips a `w == 0` corner-case panic in its
    GKR sumcheck on x86_64 hosts.

    leanVM itself sets the same flag in its workspace `.cargo/config.toml`
    — that config doesn't apply when leanVM is consumed as a git dep, so
    we set it here instead.

    `features` selects the API mode. When set, the default feature is
    suppressed (`--no-default-features`) so the caller's choice is the
    only one active.
    """
    print("Building lean-bench-workloads (release)...", flush=True)
    env = {**os.environ, "RUSTFLAGS": "-C target-cpu=native"}
    cargo_args = ["cargo", "build", "--release", "--bin", "lean-bench-workloads",
                  "--manifest-path", str(RUST_DIR / "Cargo.toml")]
    if features:
        cargo_args += ["--no-default-features", "--features", features]
    r = subprocess.run(cargo_args, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit("cargo build failed")
    return RUST_DIR / "target" / "release" / "lean-bench-workloads"


def run_workload(
    binary: Path, cli_args: list[str], samples: int, warmup: int,
) -> tuple[dict, dict, dict] | None:
    """Run one workload subprocess, sample resources, return (record, shas, branches).

    `shas` is `{"leansig_sha", "leanmultisig_sha"}` and `branches` is
    `{"leansig_branch", "leanmultisig_branch"}`, both extracted from the
    Rust output. Caller uses these to populate `toolchain.git_shas` and
    `toolchain.branches` once per run; the values are constants baked
    into the binary so any successful run has the authoritative pair.
    Returns None on failure.
    """
    proc = subprocess.Popen(
        [str(binary), *cli_args,
         "--samples", str(samples),
         "--warmup", str(warmup)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Delay sampler start slightly so we don't include the transient of
    # process startup. Rust binary loads dep crates, initializes etc.
    sampler = ResourceSampler(proc.pid, interval_ms=100)
    stdout_b, stderr_b = proc.communicate()
    resources = sampler.stop()

    if proc.returncode != 0:
        print(f"[error] {' '.join(cli_args)} exited {proc.returncode}")
        print(stderr_b.decode(errors="replace"), file=sys.stderr)
        return None

    stdout = stdout_b.decode(errors="replace").strip()
    # Take the last line — stray cargo / tracing output may appear above.
    last_line = stdout.splitlines()[-1] if stdout else ""
    try:
        rec = json.loads(last_line)
    except json.JSONDecodeError as e:
        print(f"[error] {' '.join(cli_args)}: could not parse runner JSON: {e}\nstdout: {stdout!r}",
              file=sys.stderr)
        return None

    samples_ns = rec.get("samples_ns", [])
    summary = _summarize(samples_ns)

    record = {
        "name": rec["workload"],
        "unit": rec.get("unit", "ns"),
        "samples_ns": samples_ns,              # keep raw samples for charts
        "timing": summary,
        "resources": resources,
        "meta": rec.get("meta", {}),
    }
    shas = {
        "leansig_sha": rec.get("leansig_sha", "unknown"),
        "leanmultisig_sha": rec.get("leanmultisig_sha", "unknown"),
    }
    branches = {
        "leansig_branch": rec.get("leansig_branch", "unknown"),
        "leanmultisig_branch": rec.get("leanmultisig_branch", "unknown"),
    }
    return record, shas, branches


def _summarize(samples: list[int]) -> dict:
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / max(n - 1, 1)
    return {
        "n": n,
        "mean_ns": int(mean),
        "stddev_ns": int(math.sqrt(var)),
        "min_ns": int(s[0]),
        "p5_ns":  int(s[max(0, int(n * 0.05))]),
        "p50_ns": int(s[n // 2]),
        "p95_ns": int(s[max(0, int(n * 0.95) - 1)]),
        "max_ns": int(s[-1]),
    }


def main():
    args = parse_args()
    binary = build_runner(features=args.features)

    machine = sysinfo.capture()
    label = args.label or sysinfo.auto_label()
    machine["label"] = label

    if args.label is None:
        print(f"Label (auto): {label}  (override with --label)")
    print(f"Machine: {machine['cpu_model']} ({machine['fingerprint']})")
    print(f"         {machine['physical_cores']} physical / "
          f"{machine['logical_cores']} logical cores, {machine['memory_gb']} GB RAM")
    print(f"         {machine['os']}")
    print()

    workloads_to_run = select_workloads(args)
    print(f"Running {len(workloads_to_run)} workload(s): "
          f"{', '.join(w.name for w in workloads_to_run)}")
    print()

    workload_results = []
    shas: dict = {"leansig_sha": "unknown", "leanmultisig_sha": "unknown"}
    branches: dict = {"leansig_branch": "unknown", "leanmultisig_branch": "unknown"}
    for w in workloads_to_run:
        n = w.samples_override if w.samples_override is not None else args.samples
        suffix = f" (n={n})" if w.samples_override is not None else ""
        print(f"  → {w.name}{suffix} ...", end="", flush=True)
        result = run_workload(binary, w.cli_args, n, args.warmup)
        if result is not None:
            rec, run_shas, run_branches = result
            mean_ms = rec["timing"]["mean_ns"] / 1e6
            print(f" {mean_ms:.2f} ms (n={rec['timing']['n']})")
            workload_results.append(rec)
            # SHAs and branches are baked-in constants identical across
            # every run, so any successful workload supplies the
            # authoritative pair.
            if shas["leansig_sha"] == "unknown":
                shas = run_shas
                branches = run_branches
        else:
            print(" FAILED")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{timestamp}__{machine['fingerprint']}"

    record = {
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine": machine,
        "toolchain": {
            **sysinfo.toolchain(),
            "git_shas": shas,
            "branches": branches,
        },
        "workloads": workload_results,
        "notes": args.notes,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
