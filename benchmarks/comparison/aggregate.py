"""Aggregate one or more ``comparison_v1.json`` files into a markdown report.

Per-workload ranked table (impls sorted by images_per_sec, with
segpaste's speedup vs each reference) plus a cross-workload pivot.
Footer carries the env block and a skip/error summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _load(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


def _fmt_throughput(impl: dict[str, Any]) -> str:
    if impl["status"] != "ok":
        return impl.get("skip_reason") or impl.get("error_message") or impl["status"]
    ips = impl.get("images_per_sec")
    bps = impl.get("batches_per_sec")
    if ips is None or bps is None:
        return "—"
    return f"{ips:,.1f} img/s · {bps:,.2f} bat/s"


def _fmt_ms(impl: dict[str, Any]) -> str:
    if impl["status"] != "ok":
        return "—"
    report = impl.get("report") or {}
    median_ns = report.get("median_ns")
    iqr_over_median = report.get("iqr_over_median")
    if not median_ns:
        return "—"
    return f"{median_ns / 1e6:,.2f} ms (IQR/med {iqr_over_median:.0%})"


def _ranked_table(report: dict[str, Any]) -> str:
    impls = report["implementations"]
    ok = [i for i in impls if i["status"] == "ok"]
    ok.sort(key=lambda i: -(i.get("images_per_sec") or 0.0))
    skipped = [i for i in impls if i["status"] != "ok"]

    rows = [
        "| rank | impl | throughput | latency | vs segpaste |",
        "| ---: | :--- | :--- | :--- | ---: |",
    ]
    segpaste_ips = next(
        (i.get("images_per_sec") for i in ok if i["name"] == "segpaste"), None
    )
    for rank, impl in enumerate(ok, start=1):
        ips = impl.get("images_per_sec")
        vs = "—"
        if (
            impl["name"] != "segpaste"
            and segpaste_ips is not None
            and ips is not None
            and ips > 0
        ):
            vs = f"{segpaste_ips / ips:.2f}x"
        elif impl["name"] == "segpaste":
            vs = "1.00x (anchor)"
        rows.append(
            f"| {rank} | `{impl['name']}` | {_fmt_throughput(impl)} "
            f"| {_fmt_ms(impl)} | {vs} |"
        )
    for impl in skipped:
        rows.append(f"| — | `{impl['name']}` | {_fmt_throughput(impl)} | — | — |")
    return "\n".join(rows)


def _workload_heading(report: dict[str, Any]) -> str:
    w = report["workload"]
    return (
        f"### `B={w['batch_size']} · "
        f"{w['image_size']}² · k=[{w['k_range'][0]}, {w['k_range'][1]}] · "
        f"{w['device']}`"
    )


def _env_block(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return ""
    env = reports[0]["env"]
    return (
        "## Environment\n\n"
        f"- Python `{env['python']}`, torch `{env['torch']}`, "
        f"torchvision `{env['torchvision']}`, numpy `{env['numpy']}`\n"
        f"- mmdet `{env['mmdet']}`, segpaste `{env['segpaste_version']}`\n"
        f"- CPU: `{env['cpu_model'] or 'unknown'}`\n"
        f"- Runner: `{env['runner']}` · commit `{env['commit_sha']}`\n"
        f"- Captured: `{env['timestamp']}`\n"
    )


def _summary(reports: list[dict[str, Any]]) -> str:
    total = sum(len(r["implementations"]) for r in reports)
    ok = sum(1 for r in reports for i in r["implementations"] if i["status"] == "ok")
    skipped = sum(
        1 for r in reports for i in r["implementations"] if i["status"] == "skipped"
    )
    errored = sum(
        1 for r in reports for i in r["implementations"] if i["status"] == "error"
    )
    return (
        f"**{ok} ok · {skipped} skipped · {errored} errored** "
        f"across {len(reports)} workloads ({total} cells)."
    )


def render(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "_No reports loaded._\n"
    out: list[str] = [
        "# Throughput comparison (ADR-0016)",
        "",
        _summary(reports),
        "",
        "## Per-workload",
        "",
    ]
    for report in reports:
        out.append(_workload_heading(report))
        out.append("")
        out.append(_ranked_table(report))
        out.append("")
    out.append(_env_block(reports))
    out.append(
        "\n## Caveats\n\n"
        "- Throughput is measured at `num_threads=1` (matches ADR-0002). "
        "Wall-clock at fixed thread, not throughput-per-core.\n"
        "- Adapter cost is excluded from the timed window per ADR-0016 §2. "
        "Implementations are not semantically equivalent — they apply "
        'different randomness; the comparison is "cost of one batch of '
        "augmentation under each impl's own semantics.\"\n"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reports = _load(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(reports))
    print(f"wrote {args.out}  reports={len(reports)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
