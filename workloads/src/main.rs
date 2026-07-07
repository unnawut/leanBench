//! lean-bench-workloads — single binary that runs one workload per invocation
//! and prints a JSON record of per-sample timings on stdout.
//!
//! The Python orchestrator wraps each invocation to sample CPU and memory,
//! then merges everything into one committed run file. Per-invocation
//! process isolation means resource stats cover only that workload's work,
//! not the union.

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use serde::Serialize;

mod workloads;

const LEANSIG_SHA: &str = match option_env!("LEANSIG_SHA") {
    Some(s) => s,
    None => "unknown",
};
// `LEANMULTISIG_*` is the historical schema label — leanMultisig was renamed
// to leanVM upstream (2026-06) but the env var / JSON key names are kept so
// every existing `results/*.json` keeps deserializing through one parser path.
// UI surfaces are relabeled to "leanVM" without touching the on-disk schema.
const LEANMULTISIG_SHA: &str = match option_env!("LEANMULTISIG_SHA") {
    Some(s) => s,
    None => "unknown",
};
const LEANSIG_BRANCH: &str = match option_env!("LEANSIG_BRANCH") {
    Some(s) => s,
    None => "unknown",
};
const LEANMULTISIG_BRANCH: &str = match option_env!("LEANMULTISIG_BRANCH") {
    Some(s) => s,
    None => "unknown",
};

#[derive(Parser)]
#[command(version, about = "Run one leanSig / leanMultisig workload and emit JSON samples.")]
enum Cli {
    #[command(name = "aggregate-flat",
              about = "Aggregation: flat n-sig leaf at LOG_INV_RATE_PROD=2")]
    AggregateFlat {
        /// Number of raw XMSS signatures to aggregate.
        n: usize,
        #[command(flatten)]
        common: CommonArgs,
    },
    #[command(name = "aggregate-tree",
              about = "Aggregation: fan-to-1 recursion over fan n-sig leaves at LOG_INV_RATE_PROD=2")]
    AggregateTree {
        /// Fan-in at the root (e.g. 2, 4, 8).
        fan: usize,
        /// Raw XMSS signatures per leaf.
        n: usize,
        #[command(flatten)]
        common: CommonArgs,
    },

    #[command(name = "aggregate-deep",
              about = "Aggregation: root_fan-to-1 over mid_fan-to-1 recursion over n-sig leaves at LOG_INV_RATE_PROD=2")]
    AggregateDeep {
        /// Fan-in at the root (combines this many mid-nodes).
        root_fan: usize,
        /// Fan-in at each mid-node (combines this many leaves).
        mid_fan: usize,
        /// Raw XMSS signatures per leaf.
        n: usize,
        #[command(flatten)]
        common: CommonArgs,
    },

    #[command(name = "split",
              about = "split_type_2 at index 0 of a K-component, N-per-component parent type-2 at LOG_INV_RATE_PROD=2")]
    Split {
        /// Raw XMSS signers per component (N).
        per_component: usize,
        /// Number of components in the parent type-2 (K).
        n_components: usize,
        #[command(flatten)]
        common: CommonArgs,
    },
    #[command(name = "merge-split-and-original",
              about = "merge_many_type_1 over one split-derived type-1 and one freshly-aggregated original")]
    MergeSplitAndOriginal {
        /// Raw XMSS signers per component (N).
        per_component: usize,
        /// Number of components in the parent type-2 (K).
        n_components: usize,
        #[command(flatten)]
        common: CommonArgs,
    },
    #[command(name = "merge-split-and-leaves",
              about = "aggregate_type_1 over one split-derived type-1 plus n_new_leaves fresh raw XMSS signatures")]
    MergeSplitAndLeaves {
        /// Raw XMSS signers per component in the parent type-2 (N).
        per_component: usize,
        /// Number of components in the parent type-2 (K).
        n_components: usize,
        /// Fresh raw XMSS signatures to mix in alongside the split (L).
        n_new_leaves: usize,
        #[command(flatten)]
        common: CommonArgs,
    },
}

#[derive(Parser, Clone)]
struct CommonArgs {
    /// Number of timed samples.
    #[arg(long, default_value_t = 30)]
    samples: usize,
    /// Warm-up iterations discarded before timing.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Deterministic seed.
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,
}

#[derive(Serialize)]
struct Record {
    workload: String,
    unit: &'static str,
    samples_ns: Vec<u128>,
    warmup: usize,
    meta: serde_json::Value,
    leansig_sha: &'static str,
    leanmultisig_sha: &'static str,
    leansig_branch: &'static str,
    leanmultisig_branch: &'static str,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let rec = match cli {
        Cli::AggregateFlat { n, common } => workloads::aggregate::flat_r2(&common, n),
        Cli::AggregateTree { fan, n, common } => workloads::aggregate::tree_r2(&common, fan, n),
        Cli::AggregateDeep { root_fan, mid_fan, n, common } =>
            workloads::aggregate::deep_r2(&common, root_fan, mid_fan, n),
        Cli::Split { per_component, n_components, common } =>
            workloads::aggregate::split_r2(&common, per_component, n_components),
        Cli::MergeSplitAndOriginal { per_component, n_components, common } =>
            workloads::aggregate::merge_split_and_original_r2(&common, per_component, n_components),
        Cli::MergeSplitAndLeaves { per_component, n_components, n_new_leaves, common } =>
            workloads::aggregate::merge_split_and_leaves_r2(&common, per_component, n_components, n_new_leaves),
    }?;

    let out = serde_json::to_string(&rec)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(out.as_bytes())?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn make_record(
    workload: &str,
    samples_ns: Vec<u128>,
    warmup: usize,
    meta: serde_json::Value,
) -> Record {
    Record {
        workload: workload.into(),
        unit: "ns",
        samples_ns,
        warmup,
        meta,
        leansig_sha: LEANSIG_SHA,
        leanmultisig_sha: LEANMULTISIG_SHA,
        leansig_branch: LEANSIG_BRANCH,
        leanmultisig_branch: LEANMULTISIG_BRANCH,
    }
}
