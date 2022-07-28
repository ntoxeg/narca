import logging
import subprocess
from typing import Any, Optional

from icecream import ic

from .narsese import *

logger = logging.getLogger(__name__)


def send_input(process: subprocess.Popen, input_: str) -> None:
    """Send input to NARS server"""
    stdin, stdout = process.stdin, process.stdout
    if stdin is None or stdout is None:
        raise RuntimeError("Process has no stdin or stdout.")

    stdin.write(input_ + "\n")
    stdin.flush()


def get_raw_output(process: subprocess.Popen) -> list[str]:
    """Get raw output from NARS server"""
    stdin, stdout = process.stdin, process.stdout
    if stdin is None or stdout is None:
        raise RuntimeError("Process has no stdin or stdout.")

    stdin.write("0\n")
    stdin.flush()
    ret: str = ""
    before: list[str] = []
    while "done with 0 additional inference steps." != ret.strip():
        if ret != "":
            before.append(ret.strip())
        ret = stdout.readline()
    logger.debug("\n".join(before))
    return before[:-1]


def get_output(process: subprocess.Popen) -> dict[str, Any]:
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
    process: subprocess.Popen,
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
            return None
        patience -= 1

        if goal_reentry is not None:
            send_input(process, nal_demand(goal_reentry.symbol))

        send_input(process, str(think_ticks))
        output = get_output(process)
    # ic("Got a matching output.")
    return output


def setup_nars_ops(
    process: subprocess.Popen, ops: list[str], babblingops: Optional[int] = None
):
    """Setup NARS operations"""
    for i, op in enumerate(ops):
        send_input(process, f"*setopname {i+1} {op}")
    if babblingops is None:
        send_input(process, f"*babblingops={len(ops)}")
    else:
        send_input(process, f"*babblingops={babblingops}")


def setup_nars(
    process: subprocess.Popen,
    ops: list[str],
    motor_babbling: Optional[float] = None,
    babblingops: Optional[int] = None,
    volume: Optional[int] = None,
    decision_threshold: Optional[float] = None,
):
    """Send NARS settings"""
    send_input(process, "*reset")
    setup_nars_ops(process, ops, babblingops=babblingops)
    if motor_babbling is not None:
        send_input(process, f"*motorbabbling={motor_babbling}")
    if volume is not None:
        send_input(process, f"*volume={volume}")
    if decision_threshold is not None:
        send_input(process, f"*decisionthreshold={decision_threshold}")
