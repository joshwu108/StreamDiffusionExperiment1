"""
Opt-in per-iteration stage timer for the StreamDiffusionV2 inference loops (plan.md Step 3).

Usage inside a rank loop:

    timer = StageTimer(rank, out_dir, enabled=out_dir is not None)
    ...
    timer.mark("iter_start")          # top of the loop body
    ... receive ...
    timer.mark("recv")                # interval iter_start -> recv is the "recv" stage
    ... DiT ...
    timer.mark("dit")
    ...
    torch.cuda.synchronize()          # the loop's own sync
    timer.end_iter(iteration=..., frames=..., ...)

Each `mark` records a CUDA event on the *current* stream plus a host perf_counter; neither adds a
dependency or a host sync. `end_iter` must run after every recorded event has completed (the loops
already synchronize at the end of each iteration) and turns consecutive marks into one row:

    {"rank": r, "iteration": k, ..., "iter_host_ms": ..., "stages": {"recv": {"gpu_ms": ..., "host_ms": ...}, ...}}

Call `mark` outside any `with torch.cuda.stream(com_stream)` block so the events land on the
compute stream. With enabled=False every method is a no-op.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch


class StageTimer:
    def __init__(self, rank: int, out_dir: Optional[str], enabled: bool = True):
        self.rank = rank
        self.out_dir = out_dir
        self.enabled = bool(enabled and out_dir)
        self._marks: List[tuple] = []          # (name, cuda event, perf_counter)
        self.rows: List[Dict[str, Any]] = []
        self.notes: List[Dict[str, Any]] = []
        self.wall_start = time.perf_counter()
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ recording
    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._marks.append((name, ev, time.perf_counter()))

    def end_iter(self, **meta: Any) -> None:
        """Close the current iteration. Must be called after the events have completed."""
        if not self.enabled:
            return
        marks = self._marks
        self._marks = []
        if len(marks) < 2:
            return
        for name, ev, _ in marks:
            assert ev.query(), f"StageTimer: event '{name}' not complete at end_iter (rank {self.rank})"
        stages: Dict[str, Dict[str, float]] = {}
        for (n0, e0, t0), (n1, e1, t1) in zip(marks[:-1], marks[1:]):
            stages[n1] = {"gpu_ms": e0.elapsed_time(e1), "host_ms": (t1 - t0) * 1e3}
        row: Dict[str, Any] = {
            "rank": self.rank,
            "t_wall_s": marks[0][2] - self.wall_start,
            "iter_host_ms": (marks[-1][2] - marks[0][2]) * 1e3,
            "iter_gpu_ms": marks[0][1].elapsed_time(marks[-1][1]),
        }
        row.update(meta)
        row["stages"] = stages
        self.rows.append(row)

    def note(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        d = {"kind": kind, "rank": self.rank, "t_wall_s": time.perf_counter() - self.wall_start}
        d.update(data)
        self.notes.append(d)

    # ------------------------------------------------------------------ output
    def dump(self) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.out_dir, f"timing_rank{self.rank}.jsonl")
        with open(path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(self.out_dir, f"schedule_rank{self.rank}.json"), "w") as f:
            json.dump(self.notes, f, indent=1)
