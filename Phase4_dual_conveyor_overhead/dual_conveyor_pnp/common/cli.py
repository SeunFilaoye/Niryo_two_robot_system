"""Shared argument parsing for the numbered scripts."""

from __future__ import annotations

import argparse

from common.config import check_robot_owns_conveyor, conveyor_key, robot_key


def robot_parser(description: str, conveyor: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--robot", required=True, help="150")
    if conveyor:
        parser.add_argument("--conveyor", required=True, help="C1 or C2")
    return parser


def parse_robot(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    args.robot = robot_key(args.robot)
    if hasattr(args, "conveyor"):
        args.conveyor = conveyor_key(args.conveyor)
        check_robot_owns_conveyor(args.robot, args.conveyor)
    return args


def confirm(prompt: str) -> bool:
    return input(f"{prompt} Type YES to continue: ").strip() == "YES"
