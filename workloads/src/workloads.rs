//! Workload implementations. Each returns a Record with samples_ns and a
//! workload-specific metadata blob.

use anyhow::Result;

use crate::{make_record, CommonArgs, Record};

pub mod aggregate {
    use super::*;
    #[cfg(feature = "api-leansig")]
    use ::rec_aggregation::benchmark::{run_aggregation_benchmark, AggregationTopology, BenchmarkReport};
    #[cfg(feature = "api-xmss")]
    use ::rec_aggregation_main::benchmark::{run_aggregation_benchmark, AggregationTopology, BenchmarkReport};

    /// Per-node entry for the JSON `proof_kib_by_path` field.
    /// `path = []` is the root; deeper paths are the children/leaves.
    /// `depth` is convenience metadata (path length) so consumers don't have
    /// to rederive it.
    #[derive(serde::Serialize)]
    struct ProofSizeEntry {
        path: Vec<usize>,
        depth: usize,
        kib: usize,
    }

    /// Run the timed sample loop and return (samples_ns, proof_sizes_per_node, raw_reports).
    ///
    /// Proof sizes are deterministic for a given topology, so we surface them
    /// once as a flat per-path list (for the index summary) — the same data
    /// is also available inside `reports[i].nodes[j].stats.proof_kib` if
    /// callers want it from the raw stream.
    ///
    /// The full `Vec<BenchmarkReport>` is also kept and dumped into the JSON
    /// record so we never have to re-bench just to extract a metric we
    /// happened not to surface — `time_secs`, `cycles`, `memory`,
    /// `poseidons`, `dots`, `n_xmss` per node per iteration are all
    /// recoverable from the result file.
    ///
    /// `silent=true` suppresses leanVM's ANSI render so the only thing
    /// the runner prints is its own one-line JSON record.
    fn run_loop(args: &CommonArgs, topology: &AggregationTopology)
        -> (Vec<u128>, Vec<ProofSizeEntry>, Vec<BenchmarkReport>)
    {
        let mut samples = Vec::with_capacity(args.samples);
        let mut proof_sizes: Vec<ProofSizeEntry> = Vec::new();
        let mut reports: Vec<BenchmarkReport> = Vec::with_capacity(args.samples);
        for i in 0..(args.samples + args.warmup) {
            let t = std::time::Instant::now();
            let report = run_aggregation_benchmark(topology, false, true, 1);
            if i >= args.warmup {
                samples.push(t.elapsed().as_nanos());
                if proof_sizes.is_empty() {
                    proof_sizes = report.nodes.iter()
                        .map(|n| ProofSizeEntry {
                            path: n.path.clone(),
                            depth: n.path.len(),
                            kib: n.stats.proof_kib,
                        })
                        .collect();
                }
                reports.push(report);
            }
        }
        (samples, proof_sizes, reports)
    }

    /// Pull out the root and (assumed-uniform) leaf proof sizes for the
    /// summary fields. Mid-tier sizes can be read off `proof_kib_by_path`
    /// when needed; we expose root + leaf as scalars because they're the
    /// two values most analyses care about (root → published proof,
    /// leaf → safe-target proof).
    fn root_and_leaf_kib(entries: &[ProofSizeEntry]) -> (Option<usize>, Option<usize>) {
        let root = entries.iter().find(|e| e.depth == 0).map(|e| e.kib);
        let leaf = entries.iter().map(|e| e.depth).max()
            .and_then(|d| entries.iter().find(|e| e.depth == d).map(|e| e.kib));
        (root, leaf)
    }

    /// One leaf aggregator over `n` raw XMSS signatures at LOG_INV_RATE_PROD=2.
    /// Aggregation internally does heavy one-time setup (DFT twiddles, bytecode,
    /// signer cache). First call amortises it, so the warmup iterations matter
    /// — we count only post-warmup samples.
    pub fn flat_r2(args: &CommonArgs, n: usize) -> Result<Record> {
        let topology = AggregationTopology { raw_xmss: n, children: vec![], log_inv_rate: 2, overlap: 0 };
        let (samples, proof_sizes, reports) = run_loop(args, &topology);
        let (root_kib, leaf_kib) = root_and_leaf_kib(&proof_sizes);
        Ok(make_record(
            &format!("aggregate.flat_{n}_r2"),
            samples,
            args.warmup,
            serde_json::json!({
                "raw_xmss": n,
                "log_inv_rate": 2,
                "topology": "flat",
                "proof_kib_root": root_kib,
                "proof_kib_leaf": leaf_kib,
                "proof_kib_by_path": proof_sizes,
                "reports": reports,
            }),
        ))
    }

    /// `fan`-to-1 recursion: root combines `fan` `n`-sig leaves at LOG_INV_RATE_PROD=2.
    /// Reports total wall time including all leaves + the recursion step.
    /// Subtract `fan × aggregate.flat_<n>_r2` for the recursion-only cost.
    pub fn tree_r2(args: &CommonArgs, fan: usize, n: usize) -> Result<Record> {
        let leaf = AggregationTopology { raw_xmss: n, children: vec![], log_inv_rate: 2, overlap: 0 };
        let topology = AggregationTopology {
            raw_xmss: 0,
            children: vec![leaf; fan],
            log_inv_rate: 2,
            overlap: 0,
        };
        let (samples, proof_sizes, reports) = run_loop(args, &topology);
        let (root_kib, leaf_kib) = root_and_leaf_kib(&proof_sizes);
        Ok(make_record(
            &format!("aggregate.tree_{fan}x{n}_r2"),
            samples,
            args.warmup,
            serde_json::json!({
                "leaf_raw_xmss": n,
                "fan_in": fan,
                "log_inv_rate": 2,
                "topology": format!("{fan}-to-1 recursion"),
                "proof_kib_root": root_kib,
                "proof_kib_leaf": leaf_kib,
                "proof_kib_by_path": proof_sizes,
                "reports": reports,
                "note": format!("recursion-only time = root node `time_secs` from any report; or total - {fan} × aggregate.flat_{n}_r2 as a fallback"),
            }),
        ))
    }

    /// Three-level recursion: root combines `root_fan` mid-nodes, each mid-node
    /// combines `mid_fan` `n`-sig leaves, all at LOG_INV_RATE_PROD=2. Total raw
    /// signatures = root_fan × mid_fan × n. Probes the aggregate-fast-aggregate-often
    /// regime (deep recursion) against the wide 2-level `tree_*` runner.
    pub fn deep_r2(args: &CommonArgs, root_fan: usize, mid_fan: usize, n: usize) -> Result<Record> {
        let leaf = AggregationTopology { raw_xmss: n, children: vec![], log_inv_rate: 2, overlap: 0 };
        let mid = AggregationTopology {
            raw_xmss: 0,
            children: vec![leaf; mid_fan],
            log_inv_rate: 2,
            overlap: 0,
        };
        let topology = AggregationTopology {
            raw_xmss: 0,
            children: vec![mid; root_fan],
            log_inv_rate: 2,
            overlap: 0,
        };
        let (samples, proof_sizes, reports) = run_loop(args, &topology);
        let (root_kib, leaf_kib) = root_and_leaf_kib(&proof_sizes);
        Ok(make_record(
            &format!("aggregate.deep_{root_fan}x{mid_fan}x{n}_r2"),
            samples,
            args.warmup,
            serde_json::json!({
                "leaf_raw_xmss": n,
                "root_fan_in": root_fan,
                "mid_fan_in": mid_fan,
                "total_raw_xmss": root_fan * mid_fan * n,
                "log_inv_rate": 2,
                "topology": format!("{root_fan}-to-1 over {mid_fan}-to-1 recursion"),
                "proof_kib_root": root_kib,
                "proof_kib_leaf": leaf_kib,
                "proof_kib_by_path": proof_sizes,
                "reports": reports,
                "note": "per-node cost is each node's `time_secs` in reports: root is path=[], mids are path=[i], leaves are path=[i,j]",
            }),
        ))
    }

    // split / merge_split_* runners exist only on devnet5 (api-leansig with a
    // devnet5 pin) — both devnet4 and main expose only run_aggregation_benchmark.
    // Stubbed here so the runner binary compiles regardless of which pin is in
    // play; the Python orchestrator gates these out of the default workload set.
    pub fn split_r2(_args: &CommonArgs, _per_component: usize, _n_components: usize) -> Result<Record> {
        anyhow::bail!("split workload not available on this leanVM pin")
    }

    pub fn merge_split_and_original_r2(_args: &CommonArgs, _per_component: usize, _n_components: usize) -> Result<Record> {
        anyhow::bail!("merge_split_and_original workload not available on this leanVM pin")
    }

    pub fn merge_split_and_leaves_r2(
        _args: &CommonArgs,
        _per_component: usize,
        _n_components: usize,
        _n_new_leaves: usize,
    ) -> Result<Record> {
        anyhow::bail!("merge_split_and_leaves workload not available on this leanVM pin")
    }
}
