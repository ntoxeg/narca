import logging
from typing import Any, Optional

import pexpect
from icecream import ic

from .narsese import *

logger = logging.getLogger(__name__)


def send_input(process: pexpect.spawn, input_: str) -> None:
    """Send input to NARS server"""
    process.sendline(input_)


def get_raw_output(process: pexpect.spawn) -> list[str]:
    """Get raw output from NARS server"""
    process.sendline("0")
    process.expect(["done with 0 additional inference steps.", pexpect.EOF])
    output = [s.strip().decode("utf-8") for s in process.before.split(b"\n")][2:-3]  # type: ignore
    logger.debug("\n".join(output))
    return output


def get_output(process: pexpect.spawn) -> dict[str, Any]:
    lines = get_raw_output(process)
    executions = [parse_execution(l) for l in lines if l.startswith("^")]
    inputs = [
        parse_task(l.split("Input: ")[1]) for l in lines if l.startswith("Input:")
    ]
    derivations = [
        parse_task(l.split("Derived: " if l.startswith("Derived:") else "Revised:")[1])
        for l in lines
        if l.startswith("Derived:") or l.startswith("Revised:")
    ]
    answers = [
        parse_task(l.split("Answer: ")[1]) for l in lines if l.startswith("Answer:")
    ]
    reason = parse_reason("\n".join(lines))
    return {
        "input": inputs,
        "derivations": derivations,
        "answers": answers,
        "executions": executions,
        "reason": reason,
        "raw": "\n".join(lines),
    }


def expect_output(
    process: pexpect.spawn,
    targets: list[str],
    think_ticks: int = 5,
    patience: int = 3,
    goal_reentry: Optional[Goal] = None,
) -> Optional[dict[str, Any]]:
    # TODO: refactor - this is basically for dealing with executions
    output = get_output(process)
    while not any(
        target in exe["operator"] for target in targets for exe in output["executions"]
    ):
        if patience <= 0:
            ic("Patience has run out, returning None.")
            return None  # type: ignore
        patience -= 1

        if goal_reentry is not None:
            process.sendline(nal_demand(goal_reentry.symbol))

        process.sendline(str(think_ticks))
        output = get_output(process)
    # ic("Got a matching output.")
    return output


def setup_nars_ops(
    process: pexpect.spawn, ops: dict[str, int], babblingops: Optional[int] = None
):
    """Setup NARS operations"""
    for op in ops:
        process.sendline(f"*setopname {ops[op]} {op}")
    if babblingops is None:
        process.sendline(f"*babblingops={len(ops)}")
    else:
        process.sendline(f"*babblingops={babblingops}")


def setup_nars(
    process: pexpect.spawn,
    ops: dict[str, int],
    motorbabbling: float = 0.05,
    babblingops: Optional[int] = None,
    volume: Optional[int] = None,
):
    """Send NARS settings"""
    process.sendline("*reset")
    setup_nars_ops(process, ops, babblingops=babblingops)
    process.sendline(f"*motorbabbling={motorbabbling}")
    if volume is not None:
        process.sendline(f"*volume={volume}")
