import logging
import socket
from typing import Any, Optional

import pexpect

from .utils import *

IP = "127.0.0.1"
PORT = 50000
SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
logger = logging.getLogger(__name__)


def send_input(socket: socket.socket, input_: str) -> None:
    """Send input to NARS server"""
    socket.sendto((input_ + "\0").encode(), (IP, PORT))


# def send_input_process(process: subprocess.Popen, input_: str):
#     """Send input to NARS process"""
#     stdin = process.stdin
#     stdin.write(input_ + "\n")


def get_raw_output(process: pexpect.spawn) -> list[str]:
    """Get raw output from NARS server"""
    # outlines = process.stdout.readlines()
    # output = "\n".join(outlines)
    # process.sendline("0")
    # HACK: use socket to send input
    send_input(SOCKET, "0")
    process.expect(["done with 0 additional inference steps.", pexpect.EOF])
    # process.expect(pexpect.EOF)
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
    sock: socket.socket,
    process: pexpect.spawn,
    targets: list[str],
    think_ticks: int = 5,
    patience: int = 10,
    goal_reentry: Optional[Goal] = None,
) -> Optional[dict[str, Any]]:
    # TODO: refactor - this is basically for dealing with executions
    output = get_output(process)
    while not any(target in exe for target in targets for exe in output["executions"]):
        if patience <= 0:
            # ic("Patience has run out, returning None.")
            return None  # type: ignore
        patience -= 1

        # ic("Output is:", output)
        # ic("Waiting for:", targets)
        # sleep(1)
        if goal_reentry is not None:
            send_input(sock, nal_demand(goal_reentry.symbol))

        send_input(sock, str(think_ticks))
        output = get_output(process)
    # ic("Got a matching output.")
    return output


def setup_nars_ops(socket: socket.socket, ops: dict[str, int]):
    """Setup NARS operations"""
    for op in ops:
        send_input(socket, f"*setopname {ops[op]} {op}")
    # send_input(socket, f"*babblingops={len(ops)}")
    send_input(socket, "*babblingops=5")


# def setup_nars_ops_process(process: subprocess.Popen):
#     """Setup NARS operations"""
#     stdin = process.stdin
#     for op in NARS_OPERATIONS:
#         stdin.write(f"*setopname {NARS_OPERATIONS[op]} {op}\n")


def setup_nars(socket: socket.socket, ops: dict[str, int]):
    """Send NARS settings"""
    send_input(socket, "*reset")
    setup_nars_ops(socket, ops)
    send_input(socket, "*motorbabbling=0.05")
    # send_input(socket, "*volume=0")


# def setup_nars_process(process: subprocess.Popen):
#     """Setup NARS process"""
#     stdin = process.stdin
#     stdin.write("*reset\n")
#     setup_nars_ops_process(process)
#     stdin.write("*motorbabbling=0.1\n")
