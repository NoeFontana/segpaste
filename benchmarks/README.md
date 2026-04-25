# benchmarks/

End-to-end throughput harness for `BatchCopyPaste`. Contract pinned in
[`docs/adrs/0002-performance-baseline.md`](../docs/adrs/0002-performance-baseline.md).
Part v of the ADR adds the GPU dispatch lane (`bench-gpu.yml` /
`bench_batch_copy_paste.py --device cuda`) — see ADR-0008 §v.

## Run locally

```bash
uv sync --group bench
uv run python -m benchmarks.bench_batch_copy_paste \
  --device cpu --warmup 100 --iters 800 --image-size 512 \
  --out /tmp/run.json
```

The harness prints a one-line summary and writes a JSON report. A full
pass on a 2-core machine takes roughly 1.5 minutes.

## Acceptance rule (ADR-0002, Part ii)

A new `baseline.json` is accepted when **three** sequential local runs
with identical arguments have `median_ns` values whose run-to-run RSD
(stddev divided by mean) is under 3%:

```
RSD = stddev(m_1, m_2, m_3) / mean(m_1, m_2, m_3) < 0.03
```

Within-run `iqr_over_median` is ~15–25% on this workload (data-dependent
branching in the collator) and is retained in the JSON only as a
distribution-shape diagnostic — it is not part of the gate.

## Regression gate

On every PR, `.github/workflows/bench.yml` runs the harness and
`benchmarks.compare`:

```bash
uv run python -m benchmarks.compare \
  --baseline benchmarks/baseline.json \
  --current current.json \
  --threshold 0.05
```

A PR fails when `(current.median_ns - baseline.median_ns) / baseline.median_ns > 0.05`.

### Label escape hatches

- `skip-perf-gate` — skip the comparison step (upload still happens).
  Reserved for baseline-refresh PRs and intentional perf-affecting
  refactors.
- `retry-on-perf-flake` — on a re-run (`run_attempt > 1`), skip the
  comparison step. First-attempt runs still enforce the gate.

## Runner note

The committed `baseline.json` **must** be captured on the same GitHub
runner type the gate runs against (`ubuntu-latest`). A dev-box baseline
will cause every PR to fail or pass for the wrong reasons. After
landing `bench.yml`, trigger the `bench-cpu-nightly` workflow manually
(`gh workflow run bench.yml`), download the three JSON reports from the
artifact, confirm RSD < 3%, and open a `skip-perf-gate` PR that
replaces `baseline.json` with one of the three.

## Refresh the baseline

1. Open a PR labeled `skip-perf-gate`.
2. Run the harness three times locally with identical args; confirm the
   Part (ii) acceptance rule (run-to-run RSD of medians < 3%).
3. Copy one of the three JSONs to `benchmarks/baseline.json`.
4. Reference `ADR-0002` in the commit message. Paste the three runs'
   `median_ns` values and the computed RSD into the PR description.
5. Only `benchmarks/baseline.json` changes — no bundled feature work.

Changing `batch_size`, `image_size`, `k_range`, `warmup`, or `iters` is
a breaking change to the contract and requires a new ADR superseding
ADR-0002.
