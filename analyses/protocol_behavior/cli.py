"""Run or resume the production protocol-behavior analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .production import (
    DEFAULT_MANIFEST,
    create_input_record,
    default_results,
    load_manifest,
    production_stages,
    run_stage,
    stage_status,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the production workflow command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--list-stages", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stage-class",
        choices=["all", "cpu", "gpu", "de", "drf-cpu", "drf-gpu"],
        default="all",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--exclude-stage", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser


def _selected_stages(args: argparse.Namespace):
    manifest = load_manifest(args.manifest)
    stages = production_stages(manifest)
    if args.stage_class == "cpu":
        stages = [stage for stage in stages if stage.kind != "drf-gpu"]
    elif args.stage_class == "gpu":
        stages = [stage for stage in stages if stage.kind == "drf-gpu"]
    elif args.stage_class != "all":
        stages = [stage for stage in stages if stage.kind == args.stage_class]
    excluded = set(args.exclude_stage)
    return [stage for stage in stages if stage.id not in excluded]


def main(argv: list[str] | None = None) -> int:
    """Run the selected production stage(s), or finalize all completed stages."""
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    output = args.output_dir or default_results(manifest)
    stages = _selected_stages(args)
    if args.list_stages:
        print("\n".join(stage.id for stage in stages))
        return 0
    if args.stage:
        record = run_stage(
            args.manifest,
            args.stage,
            output,
            workers=args.workers,
            force=args.force,
        )
        print(f"completed production stage {args.stage}: {record}")
        return 0
    if args.preflight:
        record = create_input_record(args.manifest, output)
        print(f"validated production inputs: {record}")
        return 0
    if args.status:
        import json

        print(json.dumps(stage_status(args.manifest, output), indent=2, sort_keys=True))
        return 0
    if args.finalize:
        from .render import render_and_validate

        completion = render_and_validate(args.manifest, output, baseline_results=args.baseline_results)
        print(f"rendered and validated protocol-behavior production results: {completion}")
        return 0
    for stage in stages:
        run_stage(args.manifest, stage.id, output, workers=args.workers, force=args.force)
    from .render import render_and_validate

    completion = render_and_validate(args.manifest, output, baseline_results=args.baseline_results)
    print(f"completed protocol-behavior production workflow: {completion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
