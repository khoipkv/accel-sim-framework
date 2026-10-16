#!/usr/bin/env python3

# Validate NVBit SASS traces and, optionally, cross-check them against the
# hardware counters collected by util/hw_stats/run_hw.py on the same GPU.
#
# Integrity checks (always run, no GPU needed):
#   - every kernel listed in kernelslist.g has a trace file
#   - thread-block / warp counts in the trace match the grid and block dims
#   - instructions in the trace match the tracer's own stats file
#   - the SASS binary version (e.g. 89 for L40S) is what you expect
#
# Hardware checks (--hw, needs run_hw.py output from the same node/driver):
#   - kernel launch sequence, names and grid/block dims match ncu's
#   - warp-level instruction counts match ncu's smsp__inst_executed.sum
#   - ncu's Device / CC columns match the GPU you think you traced on
#
# Example:
#   ./util/tracer_nvbit/validate_traces.py \
#       --traces hw_run/traces/device-0/12.6 \
#       --hw hw_run/device-0/12.6 --expect-sm 89 --expect-device L40S

import argparse
import csv
import glob
import lzma
import os
import re
import shutil
import struct
import subprocess
import sys

TRACE_EXTS = (".tracez", ".traceg", ".trace", ".trace.xz", ".traceg.xz")
HW_INST_METRICS = ("smsp__inst_executed.sum", "sm__inst_executed.sum")
HW_CYCLE_METRICS = ("gpc__cycles_elapsed.avg", "gpc__cycles_elapsed.max")


# --------------------------------------------------------------------------
# Trace parsing
# --------------------------------------------------------------------------


def parse_header(text):
    header = {}
    for line in text.splitlines():
        m = re.match(r"^-(.+?)\s*=\s*(.*)$", line)
        if m:
            header[m.group(1).strip()] = m.group(2).strip()
    return header


def parse_dim(value):
    # "(8,1,1)" (trace header) or "(8, 1, 1)" (ncu) -> (8, 1, 1)
    nums = re.findall(r"\d+", value or "")
    return tuple(int(n) for n in nums[:3]) if len(nums) >= 3 else None


def read_tracez(path):
    """Return (header_text, tbs, warps, insts, error) from a .tracez file.

    Layout is defined in tracer_tool/tracez_format.h: null-terminated header,
    zstd payloads, then an index footer whose offset is the last 8 bytes.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = b""
        while b"\0" not in raw:
            chunk = f.read(65536)
            if not chunk:
                return raw.decode(errors="replace"), 0, 0, 0, "no header terminator"
            raw += chunk
        header = raw.split(b"\0", 1)[0].decode(errors="replace")
        if size < 8:
            return header, 0, 0, 0, "file truncated"
        f.seek(size - 8)
        (index_offset,) = struct.unpack("<Q", f.read(8))
        if index_offset >= size - 8:
            return header, 0, 0, 0, "bad index offset (truncated file?)"
        f.seek(index_offset)
        footer = f.read(size - 8 - index_offset)
    pos = 0

    def take(fmt):
        nonlocal pos
        vals = struct.unpack_from(fmt, footer, pos)
        pos += struct.calcsize(fmt)
        return vals

    try:
        (num_tbs,) = take("<I")
        warps = insts = 0
        for _ in range(num_tbs):
            warp_count = take("<11I")[10]
            for _ in range(warp_count):
                inst_count, num_subchunks, _offset = take("<IIQ")
                sub = take("<%dI" % num_subchunks) if num_subchunks else ()
                if sum(sub) != inst_count:
                    return header, 0, 0, 0, "sub-chunk counts do not sum to warp count"
                warps += 1
                insts += inst_count
    except struct.error:
        return header, 0, 0, 0, "index footer truncated"
    if pos != len(footer):
        return header, num_tbs, warps, insts, "%d unexpected bytes after index" % (
            len(footer) - pos
        )
    return header, num_tbs, warps, insts, ""


def read_text_trace(path):
    """Same as read_tracez for the older plain-text (.traceg/.trace[.xz]) format."""
    opener = lzma.open if path.endswith(".xz") else open
    header_lines = []
    tbs = warps = insts = 0
    in_header = True
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if in_header:
                if line.startswith("#BEGIN_TB"):
                    in_header = False
                else:
                    header_lines.append(line)
                    continue
            if line.startswith("#BEGIN_TB"):
                tbs += 1
            elif line.startswith("warp ="):
                warps += 1
            elif line.startswith("insts ="):
                insts += int(line.split("=")[1])
    return "".join(header_lines), tbs, warps, insts, ""


def kernel_key(filename):
    # kernel-3-ctx_0x1ef05a0.tracez / .trace.xz / .traceg -> kernel-3-ctx_0x1ef05a0
    return re.sub(r"\.trace.*$", "", os.path.basename(filename.strip()))


def read_tracer_stats(trace_dir):
    """Parse stats_ctx_* / stats.csv written by the tracer, keyed by kernel_key."""
    stats = {}
    for path in glob.glob(os.path.join(trace_dir, "stats*")):
        with open(path, newline="") as f:
            rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
        if not rows:
            continue
        cols = {name: i for i, name in enumerate(rows[0])}
        if "kernel id" not in cols or "total_insts" not in cols:
            continue
        for r in rows[1:]:
            try:
                stats[kernel_key(r[cols["kernel id"]])] = {
                    "name": r[cols["kernel mangled name"]],
                    "total": int(r[cols["total_insts"]]),
                    "reported": int(r[cols["total_reported_insts"]])
                    if "total_reported_insts" in cols
                    else None,
                }
            except (IndexError, ValueError, KeyError):
                continue
    return stats


def find_trace_file(trace_dir, listed_name):
    path = os.path.join(trace_dir, listed_name)
    if os.path.exists(path):
        return path
    key = kernel_key(listed_name)
    for ext in TRACE_EXTS:
        if os.path.exists(os.path.join(trace_dir, key + ext)):
            return os.path.join(trace_dir, key + ext)
    return None


def check_trace_run(trace_dir, expect_sm):
    """Integrity-check one traces/ directory. Returns (kernels, problems)."""
    listed = []
    with open(os.path.join(trace_dir, "kernelslist.g")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("kernel-"):
                listed.append(line)
    stats = read_tracer_stats(trace_dir)
    kernels, problems = [], []
    for i, name in enumerate(listed):
        key = kernel_key(name)
        k = {"seq": i, "key": key, "file": name, "stats": stats.get(key)}
        path = find_trace_file(trace_dir, name)
        if path is None:
            problems.append("%s: listed in kernelslist.g but file is missing" % key)
            kernels.append(k)
            continue
        reader = read_tracez if path.endswith(".tracez") else read_text_trace
        header_text, tbs, warps, insts, err = reader(path)
        header = parse_header(header_text)
        grid, block = parse_dim(header.get("grid dim")), parse_dim(header.get("block dim"))
        k.update(
            name=header.get("kernel name", ""),
            grid=grid,
            block=block,
            binary_version=header.get("binary version"),
            trace_insts=insts,
        )
        if err:
            problems.append("%s: %s" % (key, err))
        if grid and block:
            exp_tbs = grid[0] * grid[1] * grid[2]
            exp_warps = exp_tbs * -(-(block[0] * block[1] * block[2]) // 32)
            if tbs != exp_tbs:
                problems.append("%s: %d thread blocks, grid says %d" % (key, tbs, exp_tbs))
            elif warps != exp_warps:
                problems.append("%s: %d warps, expected %d" % (key, warps, exp_warps))
        s = k["stats"]
        if s:
            if s["reported"] is not None and insts != s["reported"]:
                problems.append(
                    "%s: trace has %d insts, tracer reported %d"
                    % (key, insts, s["reported"])
                )
            if s["reported"] is not None and s["reported"] != s["total"]:
                k["note"] = "tracer dropped %d insts (spinlock/limit)" % (
                    s["total"] - s["reported"]
                )
        if expect_sm and k["binary_version"] != str(expect_sm):
            problems.append(
                "%s: binary version %s, expected %s"
                % (key, k["binary_version"], expect_sm)
            )
        kernels.append(k)
    return kernels, problems


# --------------------------------------------------------------------------
# Hardware (ncu) parsing
# --------------------------------------------------------------------------


def to_number(value):
    try:
        return float(value.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def read_ncu_csv(path):
    """Parse the `ncu --import ... --csv --page raw` dump written by run_hw.py."""
    with open(path, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    start = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if start is None:
        return None
    header = rows[start]
    col = {name: i for i, name in enumerate(header)}
    kernels = []
    for r in rows[start + 1 :]:
        if not r or not r[0].strip().isdigit() or len(r) < len(header):
            continue  # units row, profiler chatter, or a truncated line

        def get(name):
            return r[col[name]] if name in col else None

        insts = next(
            (to_number(get(m)) for m in HW_INST_METRICS if to_number(get(m)) is not None),
            None,
        )
        cycles = next(
            (to_number(get(m)) for m in HW_CYCLE_METRICS if to_number(get(m)) is not None),
            None,
        )
        # Depending on the ncu version, "Device" is either the GPU name or just
        # its index, so prefer the display-name attribute when it was dumped.
        device = (get("device__attribute_display_name") or "").strip()
        if not device and not (get("Device") or "").strip().isdigit():
            device = (get("Device") or "").strip()
        cc = (get("CC") or "").strip()
        major = get("device__attribute_compute_capability_major")
        minor = get("device__attribute_compute_capability_minor")
        if not cc and major and minor:
            cc = "%d.%d" % (to_number(major), to_number(minor))
        kernels.append(
            {
                "id": int(r[0]),
                "name": get("Kernel Name") or get("Function Name") or "",
                "grid": parse_dim(get("Grid Size")),
                "block": parse_dim(get("Block Size")),
                "insts": int(insts) if insts is not None else None,
                "cycles": cycles,
                "device": re.sub(r"\s*\(\d+\)$", "", device),
                "cc": cc,
            }
        )
    kernels.sort(key=lambda k: k["id"])
    return kernels


def find_ncu_csv(hw_run_dir):
    candidates = glob.glob(os.path.join(hw_run_dir, "*.nsight"))
    return max(candidates, key=os.path.getmtime) if candidates else None


# --------------------------------------------------------------------------
# Kernel name normalization and alignment
# --------------------------------------------------------------------------

_demangle_cache = {}


def demangle_all(names):
    todo = sorted({n for n in names if n.startswith("_Z") and n not in _demangle_cache})
    if todo and shutil.which("c++filt"):
        out = subprocess.run(
            ["c++filt"], input="\n".join(todo), capture_output=True, text=True
        ).stdout.splitlines()
        if len(out) == len(todo):
            _demangle_cache.update(zip(todo, out))
    for n in todo:
        if n not in _demangle_cache:
            m = re.match(r"_Z(\d+)(\w+)", n)  # simple non-nested names only
            _demangle_cache[n] = m.group(2)[: int(m.group(1))] if m else n


def base_name(name):
    """Reduce a mangled, demangled or ncu 'function' name to its identifier."""
    name = _demangle_cache.get(name, name).strip()
    name = name.replace("(anonymous namespace)", "anonymous_namespace")
    depth, cut = 0, len(name)
    for i, ch in enumerate(name):  # drop the parameter list
        if ch in "<":
            depth += 1
        elif ch in ">":
            depth -= 1
        elif ch == "(" and depth == 0 and i > 0:
            cut = i
            break
    name = name[:cut]
    while name.endswith(">"):  # drop trailing template arguments
        depth, i = 0, len(name) - 1
        while i >= 0:
            depth += {">": 1, "<": -1}.get(name[i], 0)
            if depth == 0:
                break
            i -= 1
        name = name[:i].rstrip()
    name = name.split()[-1] if name.split() else name  # drop the return type
    return name.split("::")[-1]


def align(trace_kernels, hw_kernels):
    """Greedily pair kernels in launch order, skipping kernels only one side has
    (e.g. a DYNAMIC_KERNEL_RANGE trace or an ncu -c limit)."""
    pairs, j = [], 0
    for t in trace_kernels:
        tname = base_name(t.get("name") or (t.get("stats") or {}).get("name", ""))
        match = None
        for k in range(j, len(hw_kernels)):
            h = hw_kernels[k]
            if base_name(h["name"]) == tname and (
                h["grid"] is None or t.get("grid") is None or h["grid"] == t["grid"]
            ):
                match = k
                break
        if match is None:
            pairs.append((t, None))
            continue
        pairs.append((t, hw_kernels[match]))
        j = match + 1
    return pairs, len(hw_kernels) - sum(1 for _, h in pairs if h is not None)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def find_trace_runs(root, app_filter):
    runs = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("data", "data_dirs")]
        if os.path.basename(dirpath) == "traces" and "kernelslist.g" in filenames:
            rel = os.path.relpath(os.path.dirname(dirpath), root)
            if not app_filter or any(a in rel for a in app_filter):
                runs.append((rel, dirpath))
    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__ or "Validate NVBit traces against tracer stats and ncu."
    )
    parser.add_argument(
        "--traces",
        required=True,
        help="trace root, e.g. hw_run/traces/device-0/12.6 (searched recursively)",
    )
    parser.add_argument(
        "--hw",
        help="run_hw.py output root with the same layout, e.g. hw_run/device-0/12.6",
    )
    parser.add_argument("--expect-sm", type=int, help="required binary version, e.g. 89")
    parser.add_argument(
        "--expect-device", help="substring the ncu Device name must contain, e.g. L40S"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="allowed |trace-hw|/hw instruction difference in percent (default 0: exact)",
    )
    parser.add_argument("-B", "--apps", help="comma-separated app name filter")
    parser.add_argument("--csv", help="write per-kernel results to this CSV file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every kernel, not just failures"
    )
    args = parser.parse_args()

    runs = find_trace_runs(args.traces, args.apps.split(",") if args.apps else None)
    if not runs:
        sys.exit("No traces/kernelslist.g found under %s" % args.traces)

    all_rows, failed_runs, devices = [], 0, set()
    for rel, trace_dir in runs:
        kernels, problems = check_trace_run(trace_dir, args.expect_sm)
        versions = sorted({k.get("binary_version") for k in kernels if k.get("binary_version")})
        trace_total = sum(k.get("trace_insts") or 0 for k in kernels)
        line = "%-60s kernels=%-4d insts=%-12s sm=%s" % (
            rel[:60],
            len(kernels),
            "{:,}".format(trace_total),
            ",".join(versions) or "?",
        )

        hw_status = ""
        hw_failed = False
        rows = []
        if args.hw:
            hw_dir = os.path.join(args.hw, rel)
            ncu_csv = find_ncu_csv(hw_dir)
            hw_kernels = read_ncu_csv(ncu_csv) if ncu_csv else None
            if not hw_kernels:
                problems.append("no ncu data (*.nsight) under %s" % hw_dir)
                hw_status = "HW: missing"
            else:
                demangle_all([k.get("name", "") for k in kernels] + [h["name"] for h in hw_kernels])
                pairs, hw_extra = align(kernels, hw_kernels)
                counts = {"EXACT": 0, "CLOSE": 0, "MISMATCH": 0, "UNMATCHED": 0}
                for h in hw_kernels:
                    if h["device"]:
                        devices.add(h["device"])
                for t, h in pairs:
                    expected = (t.get("stats") or {}).get("total", t.get("trace_insts"))
                    row = {
                        "run": rel,
                        "seq": t["seq"],
                        "kernel": t.get("name") or t["key"],
                        "trace_insts": t.get("trace_insts"),
                        "tracer_total_insts": expected,
                        "hw_insts": h["insts"] if h else None,
                        "hw_cycles": h["cycles"] if h else None,
                        "hw_device": h["device"] if h else "",
                        "status": "UNMATCHED",
                        "detail": t.get("note", ""),
                    }
                    if h is not None:
                        issues = []
                        if h["grid"] and t.get("grid") and h["grid"] != t["grid"]:
                            issues.append("grid %s vs hw %s" % (t["grid"], h["grid"]))
                        if h["block"] and t.get("block") and h["block"] != t["block"]:
                            issues.append("block %s vs hw %s" % (t["block"], h["block"]))
                        cc = h["cc"].replace(".", "")
                        if cc and t.get("binary_version") and cc != t["binary_version"]:
                            issues.append(
                                "trace sm_%s vs hw CC %s" % (t["binary_version"], h["cc"])
                            )
                        if h["insts"] is None or expected is None:
                            issues.append("no instruction count")
                            row["status"] = "MISMATCH"
                        else:
                            diff = abs(expected - h["insts"]) / max(h["insts"], 1) * 100
                            row["diff_pct"] = round(diff, 4)
                            if diff == 0 and not issues:
                                row["status"] = "EXACT"
                            elif diff <= args.tolerance and not issues:
                                row["status"] = "CLOSE"
                            else:
                                row["status"] = "MISMATCH"
                                issues.append(
                                    "insts %d vs hw %d (%.3f%%)" % (expected, h["insts"], diff)
                                )
                        row["detail"] = "; ".join(filter(None, [row["detail"]] + issues))
                    counts[row["status"]] += 1
                    rows.append(row)
                hw_status = "HW: %d exact, %d close, %d mismatch, %d unmatched" % (
                    counts["EXACT"],
                    counts["CLOSE"],
                    counts["MISMATCH"],
                    counts["UNMATCHED"],
                )
                if hw_extra:
                    hw_status += ", %d hw-only kernels" % hw_extra
                hw_failed = bool(counts["MISMATCH"] or counts["UNMATCHED"])
                if args.expect_device and not any(h["device"] for h in hw_kernels):
                    problems.append(
                        "ncu CSV has no device name column; cannot check '%s'"
                        % args.expect_device
                    )
                elif args.expect_device:
                    wrong = sorted(
                        {
                            h["device"]
                            for h in hw_kernels
                            if args.expect_device.lower() not in h["device"].lower()
                        }
                    )
                    if wrong:
                        problems.append(
                            "ncu device %s does not match '%s'"
                            % (", ".join(wrong), args.expect_device)
                        )

        failed = bool(problems) or hw_failed
        status = "FAIL" if failed else "OK"
        failed_runs += failed
        print("%-4s %s %s" % (status, line, hw_status))
        for p in problems[:10]:
            print("       - " + p)
        if len(problems) > 10:
            print("       - ... %d more" % (len(problems) - 10))
        for row in rows:
            if args.verbose or row["status"] in ("MISMATCH", "UNMATCHED"):
                print(
                    "       [%3d] %-9s %-40s trace=%-10s hw=%-10s %s"
                    % (
                        row["seq"],
                        row["status"],
                        base_name(row["kernel"])[:40],
                        row["tracer_total_insts"],
                        row["hw_insts"],
                        row["detail"],
                    )
                )
        all_rows.extend(rows)

    print()
    if devices:
        print("ncu device(s): " + ", ".join(sorted(devices)))
    print("%d/%d trace runs passed" % (len(runs) - failed_runs, len(runs)))

    if args.csv and all_rows:
        fields = [
            "run",
            "seq",
            "kernel",
            "status",
            "trace_insts",
            "tracer_total_insts",
            "hw_insts",
            "diff_pct",
            "hw_cycles",
            "hw_device",
            "detail",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print("per-kernel results written to " + args.csv)

    sys.exit(1 if failed_runs else 0)


if __name__ == "__main__":
    main()
