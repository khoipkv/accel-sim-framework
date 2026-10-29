#!/usr/bin/env python3

# Compare the DRAM command trace of one HBM-PIM channel produced by Accel-Sim
# (-gpgpu_pim_cmd_trace) with the one printed by Samsung's PIMSimulator.
#
# PIMSimulator side: stdout of ./sim with SHOW_SIM_OUTPUT=true and
# DEBUG_CMD_TRACE=true in the system ini. PIMBench prints the run without PIM
# first and the run with PIM second; pick one with --run. PIMSimulator prints
# a command when the rank receives it, tCMD (1) cycle after it was issued;
# messages printed by PIMRank (HAB/HAB_PIM accesses, register writes, CRF
# steps) carry PIMRank's clock, which Rank::update() steps one cycle earlier.
# Both are mapped back to the issue cycle that Accel-Sim prints.
#
# Commands are compared as (cycle, command, bank, column); fields PIMSimulator
# does not print for a command (bank of CRF/GRF/control writes) match
# anything. Precharge rows are not compared: PIMSimulator prints the rank's
# view of the open row, which is stale for reserved (PIM) rows. Commands
# issued in the last cycle of a run are never delivered to PIMSimulator's
# rank, so they only appear on the Accel-Sim side.
#
# Example:
#   ./util/pim/compare_cmd_trace.py --pimsim pimbench_add.log --run pim \
#       --accel sim_run/cmd_trace.txt

import argparse
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")

RANK_DELAY = 1  # tCMD: printed when the rank receives the command
PIMRANK_DELAY = 2  # PIMRank's clock runs one cycle ahead of the rank's

OUTLOG_ALL = r"ch\[(\d+)\] ra\[\d+\] bg\[\d+\] ba\[(\d+)\] ro\[\d+\] co\[(\d+)\] @(\d+)"
PATTERNS = {
    "ACT": re.compile(r"^ACTIVATE " + OUTLOG_ALL),
    "PRE": re.compile(
        r"^PRECHARGE ch\[(\d+)\] ra\[\d+\] bg\[\d+\] ba\[(\d+)\] ro\[\d+\] @(\d+)"
    ),
    "REF": re.compile(r"^REF ch\[(\d+)\] ra\[(\d+)\] @(\d+)"),
    "COL": re.compile(r"^(READ|WRITE|BANK_TO_PIM|PIM_TO_BANK) " + OUTLOG_ALL),
    "PIM": re.compile(
        r"^(READ|WRITE) ch(\d+) ra\d+ bg\d+ b(\d+) r\d+ c(\d+)\|\|.*@ (\d+)"
    ),
    "CRF": re.compile(r"^BWRITE_CRF ch\[(\d+)\] ra\[\d+\] idx\[(\d+)\] @(\d+)"),
    "GRF_A": re.compile(r"^BWRITE_GRF_A ch\[(\d+)\] ra\[\d+\] reg\[(\d+)\] @(\d+)"),
    "GRF_B": re.compile(r"^BWRITE_GRF_B ch\[(\d+)\] ra\[\d+\] reg\[(\d+)\] @(\d+)"),
    "SRF": re.compile(r"^BWRITE_SRF ch\[(\d+)\] ra\[\d+\] @(\d+)"),
    "CTRL": re.compile(r"^(HAB_PIM|HAB mode) ch\[(\d+)\] ra\[\d+\] @(\d+)"),
    "TO_HAB": re.compile(r"^HAB ch\[(\d+)\] ra\[\d+\] @(\d+)"),
    "TO_SB": re.compile(r"^SB mode ch\[(\d+)\] ra\[\d+\] @(\d+)"),
}

READS = ("READ", "BANK_TO_PIM")


def parse_pimsim(path, run, channel):
    with open(path, errors="replace") as f:
        lines = [ANSI.sub("", l.rstrip("\n")) for l in f]
    # PIMBench prints "<KERNEL> (PIM disabled)" before the run without PIM and
    # "<KERNEL> (PIM enabled)" before the run with PIM
    markers = [
        i for i, l in enumerate(lines) if re.search(r"\(PIM (enabled|disabled)\)", l)
    ]
    if markers:
        want = "enabled" if run == "pim" else "disabled"
        starts = [i for i in markers if want in lines[i]]
        if not starts:
            sys.exit("no '(PIM %s)' section in %s" % (want, path))
        begin = starts[0]
        later = [i for i in markers if i > begin]
        end = later[0] if later else len(lines)
        lines = lines[begin:end]
    cmds = []
    mode = "SB"
    for l in lines:
        for kind, regex in PATTERNS.items():
            m = regex.match(l)
            if m:
                break
        else:
            continue
        g = m.groups()
        if kind in ("TO_HAB", "TO_SB"):
            if int(g[0]) == channel:
                mode = "HAB" if kind == "TO_HAB" else "SB"
            continue
        if kind == "ACT":
            ch, cmd, t = (
                int(g[0]),
                ("ACT", int(g[1]), int(g[2])),
                int(g[3]) - RANK_DELAY,
            )
        elif kind == "PRE":
            ch, cmd, t = int(g[0]), ("PRE", int(g[1]), None), int(g[2]) - RANK_DELAY
        elif kind == "REF":
            ch, cmd, t = int(g[0]), ("REF", int(g[1]), None), int(g[2]) - RANK_DELAY
        elif kind == "COL":
            ch = int(g[1])
            cmd = ("RD" if g[0] in READS else "WR", int(g[2]), int(g[3]))
            # in SB mode the rank prints the access, otherwise PIMRank does
            t = int(g[4]) - (RANK_DELAY if mode == "SB" else PIMRANK_DELAY)
        elif kind == "PIM":
            ch = int(g[1])
            cmd = ("RD" if g[0] == "READ" else "WR", int(g[2]), int(g[3]))
            t = int(g[4]) - PIMRANK_DELAY
            # PIMRank::doPIM() prints every CRF instruction it visits, so a
            # command that takes a JUMP shows up once per visited instruction
            if ch == channel and cmds and cmds[-1] == (t,) + cmd:
                continue
        elif kind == "CRF":
            ch, cmd, t = (
                int(g[0]),
                ("WR", None, 0x4 + int(g[1])),
                int(g[2]) - PIMRANK_DELAY,
            )
        elif kind in ("GRF_A", "GRF_B"):
            base = 0x8 if kind == "GRF_A" else 0x18
            ch, cmd, t = (
                int(g[0]),
                ("WR", None, base + int(g[1])),
                int(g[2]) - PIMRANK_DELAY,
            )
        elif kind == "SRF":
            ch, cmd, t = int(g[0]), ("WR", None, 0x1), int(g[1]) - PIMRANK_DELAY
        else:  # control register write
            ch, cmd, t = int(g[1]), ("WR", None, 0x0), int(g[2]) - PIMRANK_DELAY
            if ch == channel:
                mode = "HAB_PIM" if g[0] == "HAB_PIM" else "HAB"
        if ch == channel:
            cmds.append((t,) + cmd)
    return cmds


def parse_accel(path, offset):
    cmds = []
    with open(path) as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            tok = l.split()
            t, kind = int(tok[0]) + offset, tok[1]
            if kind == "REF":
                cmds.append((t, "REF", int(tok[2][2:]), None))
                continue
            bank = int(tok[2][2:])
            col = int(tok[4][2:])
            cmds.append((t, kind, bank, None if kind == "PRE" else col))
    return cmds


def same(a, b):
    return all(x is None or y is None or x == y for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pimsim", required=True, help="PIMSimulator stdout")
    parser.add_argument(
        "--accel", required=True, help="Accel-Sim -gpgpu_pim_cmd_trace file"
    )
    parser.add_argument("--run", choices=["pim", "nonpim"], default="pim")
    parser.add_argument("--channel", type=int, default=0, help="PIMSimulator channel")
    parser.add_argument(
        "--offset", type=int, default=0, help="cycles added to the Accel-Sim cycles"
    )
    parser.add_argument("--context", type=int, default=5)
    args = parser.parse_args()

    ref = parse_pimsim(args.pimsim, args.run, args.channel)
    acc = parse_accel(args.accel, args.offset)
    n = min(len(ref), len(acc))
    for i in range(n):
        if not same(ref[i], acc[i]):
            print("MISMATCH at command %d" % i)
            lo = max(0, i - args.context)
            for j in range(lo, min(n, i + args.context + 1)):
                mark = ">>" if j == i else "  "
                print(
                    "%s %6d  PIMSimulator %-24s  Accel-Sim %s"
                    % (mark, j, ref[j], acc[j])
                )
            sys.exit(1)
    # commands issued in the very last cycle never reach PIMSimulator's rank
    if len(acc) > len(ref) and acc and all(c[0] == acc[-1][0] for c in acc[n:]):
        print(
            "note: %d command(s) issued in the last cycle are not printed by "
            "PIMSimulator" % (len(acc) - n)
        )
        acc = acc[:n]
    counts = {}
    for c in acc:
        counts[c[1]] = counts.get(c[1], 0) + 1
    if len(ref) != len(acc):
        print(
            "MISMATCH: same first %d commands, then PIMSimulator has %d and Accel-Sim %d"
            % (n, len(ref), len(acc))
        )
        sys.exit(1)
    print(
        "MATCH: %d commands (%s), last at cycle %d"
        % (
            n,
            " ".join("%s=%d" % kv for kv in sorted(counts.items())),
            acc[-1][0] if acc else 0,
        )
    )


if __name__ == "__main__":
    main()
