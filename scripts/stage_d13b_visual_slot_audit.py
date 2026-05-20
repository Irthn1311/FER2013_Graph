"""Placeholder for future D13B visual slot audit.

This script intentionally does not implement full slot visualization yet. D13B
training outputs already expose slot_attention, region assignment traces, and
slot metadata needed for a later audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="D13B visual slot audit placeholder")
    parser.add_argument("--output_dir", default="outputs/d13b_visual_slot_audit_placeholder")
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = [
        "# D13B Visual Slot Audit Placeholder",
        "",
        "Full visual slot audit is intentionally not implemented in this diagnostic pack.",
        "",
        "Future audit inputs:",
        "- trained D13B checkpoint",
        "- slot_attention per sample",
        "- region assignment maps from LocalAssignmentPool",
        "- slot projected pixel maps",
        "- manual review sheet",
        "",
        "No motif, semantic-region, or causal-evidence claim is made.",
        "",
    ]
    (out / "d13b_visual_slot_audit_todo.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote placeholder report to {out / 'd13b_visual_slot_audit_todo.md'}")


if __name__ == "__main__":
    main()
