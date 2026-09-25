#!/usr/bin/env python3
"""
Summarize the Step 3 timing runs (plan.md Step 3, hypotheses H1/H2).

Reads outputs/step3/<cfg>/timing_rank{r}.jsonl and schedule_rank{r}.json written by
streamv2v/stage_timer.py, keeps steady-state iterations, and prints:
  - per config / rank: median period and the median ms + share of each stage
  - per decoder: FPS (frames produced / wall time) on 1 GPU and 2 ranks, and the ratio
  - per --schedule_block run: what the scheduler saw and the split it chose (H2)
Writes outputs/step3/summary.csv and outputs/step3/stages.csv.

Usage:  python examples/summarize_step3.py [outputs/step3 [outputs/step4 ...]]
Several roots are read together (Step 4 variants next to their Step 3 baselines); the CSVs go to the last root.
Config names: {1,2}gpu_<vae>[_od][_oe][_sched]_r<n>  (od = --overlap_decode, oe = --overlap_encode).
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import median

STAGES_2 = ["recv", "encode", "dit", "send_wait", "send", "decode", "host_copy"]
STAGES_1 = ["encode", "dit", "decode", "host_copy"]
NUM_STEPS = 2          # --step 2 -> denoising_step_list [700, 500]
FRAMES_PER_CHUNK = 4


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_cfg(name):
    """'2gpu_taehv_od_oe_sched_r1' -> (ranks, vae, sched, rep, variant)."""
    m = re.match(r"(\d)gpu_(wan|taehv_parallel|taehv_full|taehv)(_od)?(_oe)?(_sched)?_(r\d+)$", name)
    if not m:
        return None
    variant = "".join(x.strip("_") + "+" for x in (m.group(3) or "", m.group(4) or "")).rstrip("+") or "base"
    return int(m.group(1)), m.group(2), bool(m.group(5)), m.group(6), variant


def steady(rows, ranks):
    """Drop warm-up, the rebalance iteration and its successor, and the drain row."""
    if ranks == 1:
        cut = NUM_STEPS + 3
    else:
        cut = (ranks + NUM_STEPS) * 2      # == InferencePipelineManager.schedule_step
    rows = sorted(rows, key=lambda r: r["iteration"])
    keep, skip_next = [], False
    for r in rows:
        if r["iteration"] <= cut:
            continue
        if skip_next:
            skip_next = False
            continue
        if r.get("sched_ms", 0) > 0:
            skip_next = True
            continue
        keep.append(r)
    return keep[:-1] if len(keep) > 1 else keep


def stage_ms(r, s):
    st = r["stages"].get(s)
    if st is None:
        return None
    # Always the CUDA-event interval on the compute stream: it is one consistent timeline whose
    # intervals sum to the iteration. Host intervals mislead for async stages (encode launches in
    # ~9 ms while its kernels take ~99 ms, and the next host-blocking stage then absorbs the wait).
    return st["gpu_ms"]


def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else ["outputs/step3"]
    root = roots[-1]
    cfgs = {}
    for d in sorted(p for r in roots for p in glob.glob(os.path.join(r, "*"))):
        meta = parse_cfg(os.path.basename(d))
        if meta is None or not glob.glob(os.path.join(d, "timing_rank*.jsonl")):
            continue
        ranks, vae, sched, rep, variant = meta
        per_rank = {}
        for p in sorted(glob.glob(os.path.join(d, "timing_rank*.jsonl"))):
            r = int(re.search(r"rank(\d+)", p).group(1))
            per_rank[r] = load_rows(p)
        notes = []
        for p in sorted(glob.glob(os.path.join(d, "schedule_rank*.json"))):
            notes += json.load(open(p))
        cfgs[os.path.basename(d)] = dict(ranks=ranks, vae=vae, sched=sched, rep=rep, variant=variant, rows=per_rank, notes=notes, dir=d)

    if not cfgs:
        print(f"no timing files under {root}")
        return

    # ------------------------------------------------------------ per config / rank stage table
    stage_rows, summary = [], []
    for name, c in cfgs.items():
        ranks = c["ranks"]
        final = ranks - 1
        stages = STAGES_1 if ranks == 1 else STAGES_2
        periods = {}
        for r, rows in sorted(c["rows"].items()):
            ss = steady(rows, ranks)
            if not ss:
                print(f"[warn] {name} rank {r}: no steady-state rows")
                continue
            period = median(x["iter_host_ms"] for x in ss)
            periods[r] = period
            rec = dict(cfg=name, ranks=ranks, vae=c["vae"], variant=c["variant"], sched=int(c["sched"]), rep=c["rep"], rank=r,
                       n_steady=len(ss), period_ms=round(period, 1))
            for s in stages:
                vals = [stage_ms(x, s) for x in ss if stage_ms(x, s) is not None]
                if vals:
                    m = median(vals)
                    rec[f"{s}_ms"] = round(m, 1)
                    rec[f"{s}_p90"] = round(sorted(vals)[int(0.9 * (len(vals) - 1))], 1)
                    rec[f"{s}_share"] = round(m / period, 3)
            stage_rows.append(rec)
        fs = steady(c["rows"][final], ranks)
        frames = sum(x["frames"] for x in fs)
        wall_ms = sum(x["iter_host_ms"] for x in fs)
        fps = 1e3 * frames / wall_ms if wall_ms else float("nan")
        dec = [stage_ms(x, "decode") + stage_ms(x, "host_copy") for x in fs
               if stage_ms(x, "decode") is not None and stage_ms(x, "host_copy") is not None]
        recv = [stage_ms(x, "recv") for x in fs if stage_ms(x, "recv") is not None]
        dasync = [x["decode_async_ms"] for x in fs if x.get("decode_async_ms") is not None]
        summary.append(dict(
            cfg=name, ranks=ranks, vae=c["vae"], variant=c["variant"], sched=int(c["sched"]), rep=c["rep"],
            n_steady=len(fs), frames_per_iter=round(frames / len(fs), 2) if fs else 0,
            period_ms=round(periods.get(final, float("nan")), 1),
            period_rank0_ms=round(periods.get(0, float("nan")), 1),
            fps=round(fps, 2),
            decode_plus_copy_ms=round(median(dec), 1) if dec else "",
            decode_share=round(median(dec) / periods[final], 3) if dec and final in periods else "",
            recv_wait_ms=round(median(recv), 1) if recv else "",
            recv_share=round(median(recv) / periods[final], 3) if recv and final in periods else "",
            decode_async_ms=round(median(dasync), 1) if dasync else "",
            blocks_final=str(fs[-1]["blocks"]) if fs else "",
        ))

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "stages.csv"), "w", newline="") as f:
        keys = sorted({k for r in stage_rows for k in r}, key=lambda k: (k not in ("cfg", "ranks", "vae", "variant", "sched", "rep", "rank", "n_steady", "period_ms"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(stage_rows)
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ------------------------------------------------------------ markdown
    print("\n### Final-rank period and decode share (steady state, medians)\n")
    print("| config | steady iters | frames/iter | period ms | rank-0 period ms | FPS | decode+copy ms (compute stream) | decode share | async decode ms | recv-wait ms | recv share | final blocks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['cfg']} | {s['n_steady']} | {s['frames_per_iter']} | {s['period_ms']} | {s['period_rank0_ms']} | {s['fps']} | "
              f"{s['decode_plus_copy_ms']} | {s['decode_share']} | {s['decode_async_ms']} | {s['recv_wait_ms']} | {s['recv_share']} | {s['blocks_final']} |")

    print("\n### Stage split per rank (median ms, share of that rank's period)\n")
    print("| config | rank | period | " + " | ".join(STAGES_2) + " |")
    print("|---|---|---|" + "---|" * len(STAGES_2))
    for r in stage_rows:
        cells = []
        for s in STAGES_2:
            if f"{s}_ms" in r:
                cells.append(f"{r[f'{s}_ms']} ({100 * r[f'{s}_share']:.0f}%)")
            else:
                cells.append("–")
        print(f"| {r['cfg']} | {r['rank']} | {r['period_ms']} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------ 1 -> 2 rank scaling
    print("\n### FPS by decoder and variant (mean over repeats; ratio vs 1 GPU base)\n")
    by = defaultdict(list)
    for s in summary:
        by[(s["vae"], s["ranks"], s["sched"], s["variant"])].append(s)

    def mean_of(k, field):
        v = [s[field] for s in by.get(k, []) if s[field] != ""]
        return sum(v) / len(v) if v else float("nan")
    variants = sorted({s["variant"] for s in summary}, key=lambda v: (v != "base", v))
    print("| decoder | variant | 1 GPU FPS | 2 ranks FPS (period ms) | ratio | 2 ranks + schedule FPS (period ms) | ratio |")
    print("|---|---|---|---|---|---|---|")
    for vae in ("wan", "taehv", "taehv_full", "taehv_parallel"):
        f1 = mean_of((vae, 1, 0, "base"), "fps")
        for variant in variants:
            f2, p2 = mean_of((vae, 2, 0, variant), "fps"), mean_of((vae, 2, 0, variant), "period_ms")
            f2s, p2s = mean_of((vae, 2, 1, variant), "fps"), mean_of((vae, 2, 1, variant), "period_ms")
            if f2 != f2 and f2s != f2s:
                continue
            print(f"| {vae} | {variant} | {f1:.2f} | {f2:.2f} ({p2:.0f}) | {f2 / f1:.2f}× | {f2s:.2f} ({p2s:.0f}) | {f2s / f1:.2f}× |")

    # ------------------------------------------------------------ repeat consistency
    print("\n### Repeat consistency (final-rank period, r1 vs r2)\n")
    groups = defaultdict(dict)
    for s in summary:
        groups[(s["vae"], s["ranks"], s["sched"], s["variant"])][s["rep"]] = s["period_ms"]
    for k, reps in sorted(groups.items()):
        if len(reps) >= 2:
            vals = list(reps.values())
            spread = (max(vals) - min(vals)) / min(vals)
            flag = "" if spread <= 0.03 else "  <-- >3 %, treat with care"
            print(f"- {k[0]} {k[1]}gpu sched={k[2]} {k[3]}: {reps}  spread {100 * spread:.1f} %{flag}")

    # ------------------------------------------------------------ H2 scheduler dump
    print("\n### H2: what the block scheduler saw (one entry per rank per --schedule_block run)\n")
    print("| config | rank | at iteration | t_dit_list (s) | t_total_list (s) | split before | split after | wall ms |")
    print("|---|---|---|---|---|---|---|---|")
    for name, c in cfgs.items():
        for n in c["notes"]:
            if n.get("kind") != "schedule":
                continue
            td = ", ".join(f"{x:.3f}" for x in n["t_dit_list"])
            tt = ", ".join(f"{x:.3f}" for x in n["t_total_list"])
            print(f"| {name} | {n['rank']} | {n['processed']} | {td} | {tt} | {n['block_before']} | {n['block_after']} | {n['wall_ms']:.0f} |")

    print(f"\nwrote {os.path.join(root, 'summary.csv')} and {os.path.join(root, 'stages.csv')}")


if __name__ == "__main__":
    main()
