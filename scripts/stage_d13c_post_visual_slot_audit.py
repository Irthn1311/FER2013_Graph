"""Placeholder for post-D13C visual slot audit staging.

The full post-D13C audit is intentionally not implemented in the D13C
diagnostic pack. D13C outputs should save enough tensors and metadata so the
next stage can inspect slot candidates after any SupCon diagnostic run.
"""

from __future__ import annotations

import argparse


TODO = """Post-D13C visual slot audit TODO:
- Reuse D13B visual slot audit contracts where possible.
- Require a completed D13C diagnostic run and checker PASS/WARN evidence.
- Stage slot_attention, slot_pixel_maps when available, z_image, z_proj, and sample metadata.
- Keep claims diagnostic-only: no motif discovery, no semantic-region claim, no causal evidence.
- D13C candidate cannot move forward without this audit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TODO placeholder for post-D13C visual slot audit staging."
    )
    parser.add_argument("--output_dir", default=None, help="Future D13C run output directory.")
    parser.add_argument("--print_todo", action="store_true", help="Print the audit TODO and exit.")
    args = parser.parse_args()
    print(TODO.strip())
    if args.output_dir:
        print(f"\nRequested output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
