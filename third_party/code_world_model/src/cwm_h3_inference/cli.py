"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .cache import prepare_caches, validate_cache
from .config import load_config
from .engine import generate


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cwm-h3-inference",
        description="MiniMax H3 Ref2VA inference with a fixed first frame and Retake34 continuation.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "generate"):
        command = subparsers.add_parser(name, allow_abbrev=False)
        command.add_argument("--config", required=True, help="JSON run configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, check_files=True)
    if args.command == "validate":
        _emit(
            {
                "status": "pass",
                "window_count": len(config.windows),
                "output_frames": config.output_frames,
            }
        )
        return 0
    if args.command == "prepare":
        _emit(prepare_caches(config))
        return 0
    if args.command == "generate":
        _emit(generate(config))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
