#!/bin/bash
# Run the kernels of PIMSimulator's PIMBench (GEMV 4096x4096, MUL 2M, ADD 1M,
# RELU 4M, each with and without PIM) on Accel-Sim's HBM-PIM model
# (SM80_HBM2PIM) and compare the cycle counts with Samsung's PIMSimulator.
#
# usage: run_pim_parity.sh [-p <PIMSimulator dir>] [-o <out dir>] [-c <channels>] [-t]
#   -p  PIMSimulator checkout with a built ./sim (default: $PIMSIM_ROOT);
#       without one only the Accel-Sim side runs
#   -o  output directory (default: ./pim_parity)
#   -c  channels simulated by Accel-Sim (default: 1; every channel runs the
#       same program, so the cycle count does not depend on it)
#   -t  also compare the DRAM command traces of channel 0
#       (runs a copy of PIMSimulator with SHOW_SIM_OUTPUT=true)
#
# Needs the Accel-Sim environment (source gpu-simulator/setup_environment.sh).
# The binary is $ACCELSIM_BIN, or $ACCELSIM_ROOT/bin/$ACCELSIM_CONFIG/accel-sim.out.

set -u
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIMSIM="${PIMSIM_ROOT:-}"
OUT="./pim_parity"
CHANNELS=1
TRACE=0
while getopts "p:o:c:th" opt; do
  case $opt in
    p) PIMSIM="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    c) CHANNELS="$OPTARG" ;;
    t) TRACE=1 ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
  esac
done

if [ -z "${ACCELSIM_ROOT:-}" ] || [ -z "${GPGPUSIM_ROOT:-}" ]; then
  echo "source gpu-simulator/setup_environment.sh first" >&2
  exit 1
fi
BIN="${ACCELSIM_BIN:-$ACCELSIM_ROOT/bin/${ACCELSIM_CONFIG:-release}/accel-sim.out}"
[ -x "$BIN" ] || { echo "accel-sim binary not found: $BIN" >&2; exit 1; }
CFG=SM80_HBM2PIM
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# name | PIMBench test | PIM run? | Accel-Sim command
KERNELS=(
  "gemv_pim|gemv|1|PIMKernel,gemv,out=4096,in=4096"
  "gemv_nopim|gemv|0|PIMKernel,stream,baseline=gemv,out=4096,in=4096"
  "mul_pim|mul|1|PIMKernel,mul,n=2097152"
  "mul_nopim|mul|0|PIMKernel,stream,baseline=mul,n=2097152"
  "add_pim|add|1|PIMKernel,add,n=1048576"
  "add_nopim|add|0|PIMKernel,stream,baseline=add,n=1048576"
  "relu_pim|relu|1|PIMKernel,relu,n=4194304"
  "relu_nopim|relu|0|PIMKernel,stream,baseline=relu,n=4194304"
)

# Accel-Sim runs (in parallel)
for k in "${KERNELS[@]}"; do
  IFS='|' read -r name test is_pim cmd <<< "$k"
  d="$OUT/accel-sim/$name"
  mkdir -p "$d/traces"
  echo "$cmd,sim_channels=$CHANNELS" > "$d/traces/kernelslist.g"
  cp "$GPGPUSIM_ROOT/configs/tested-cfgs/$CFG/"* "$d/"
  {
    cat "$GPGPUSIM_ROOT/configs/tested-cfgs/$CFG/gpgpusim.config"
    printf '\n# Accel-Sim Parameters\n'
    cat "$ACCELSIM_ROOT/configs/tested-cfgs/$CFG/trace.config"
    [ $TRACE -eq 1 ] && echo "-gpgpu_pim_cmd_trace $d/cmd_trace.txt"
  } > "$d/gpgpusim.config"
  (cd "$d" && "$BIN" -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1) &
done

# PIMSimulator reference
declare -A REF
if [ -n "$PIMSIM" ]; then
  [ -x "$PIMSIM/sim" ] || { echo "no PIMSimulator binary at $PIMSIM/sim" >&2; exit 1; }
  if [ $TRACE -eq 1 ]; then
    # a copy whose system ini prints every DRAM command
    P="$OUT/pimsim"
    mkdir -p "$P"
    cp "$PIMSIM/sim" "$P/" && cp -r "$PIMSIM/ini" "$P/" && cp "$PIMSIM"/system_hbm*.ini "$P/"
    sed -i 's/^SHOW_SIM_OUTPUT=.*/SHOW_SIM_OUTPUT=true/; s/^DEBUG_CMD_TRACE=.*/DEBUG_CMD_TRACE=true/' \
      "$P/system_hbm_64ch.ini"
    for test in gemv mul add relu; do
      (cd "$P" && ./sim --gtest_filter=PIMBenchFixture.$test 2>&1 |
        grep -aE 'ch\[0\]|ch0 |PIM (enabled|disabled)|Cycle :' > "$OUT/pimsim_$test.log") &
    done
  else
    for test in gemv mul add relu; do
      (cd "$PIMSIM" && ./sim --gtest_filter=PIMBenchFixture.$test > "$OUT/pimsim_$test.log" 2>&1) &
    done
  fi
fi
wait

if [ -n "$PIMSIM" ]; then
  for test in gemv mul add relu; do
    # PIMBench prints the cycles without PIM first, then with PIM
    cycles=($(grep -aoP '> Cycle : \K[0-9]+' "$OUT/pimsim_$test.log"))
    REF[${test}_nopim]=${cycles[0]:-?}
    REF[${test}_pim]=${cycles[1]:-?}
  done
fi

status=0
printf "%-12s %13s %11s %6s  %-9s %s\n" kernel PIMSimulator Accel-Sim diff selfcheck "command trace"
for k in "${KERNELS[@]}"; do
  IFS='|' read -r name test is_pim cmd <<< "$k"
  d="$OUT/accel-sim/$name"
  acc=$(grep -oP 'pim_kernel_dram_cycles = \K[0-9]+' "$d/sim.out" || echo "?")
  check=$(grep -oP 'pim_kernel_selfcheck = \K\w+' "$d/sim.out" || echo "?")
  ref="${REF[$name]:--}"
  diff="-"
  if [[ "$ref" =~ ^[0-9]+$ && "$acc" =~ ^[0-9]+$ ]]; then
    diff=$((acc - ref))
    [ $diff -ne 0 ] && status=1
  fi
  [ "$check" != "PASS" ] && status=1
  trace="-"
  if [ $TRACE -eq 1 ] && [ -n "$PIMSIM" ]; then
    run=$([ "$is_pim" = 1 ] && echo pim || echo nonpim)
    trace=$(python3 "$THIS_DIR/compare_cmd_trace.py" --pimsim "$OUT/pimsim_$test.log" \
      --run $run --accel "$d/cmd_trace.txt" | tail -1 | cut -c1-60)
    [[ "$trace" == MATCH* ]] || status=1
  fi
  printf "%-12s %13s %11s %6s  %-9s %s\n" "$name" "$ref" "$acc" "$diff" "$check" "$trace"
done
exit $status
