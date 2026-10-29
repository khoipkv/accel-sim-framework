# HBM-PIM in Accel-Sim

GPGPU-Sim's DRAM model includes a timing model of Samsung's HBM2-PIM
(processing-in-memory), ported from Samsung's PIMSimulator (a DRAMSim2
fork). Each pseudo-channel holds:
- a PIM rank with the SB / HAB / HAB_PIM mode machine;
- all-bank command semantics;
- 8 PIM blocks that step through a 32-entry CRF program on every column
  command issued in PIM mode.

The model is timing only, like PIMSimulator's `NO_STORAGE` build: no data
values are computed. PIM kernels are launched from the trace command list and
run on the DRAM channels without the SMs.

Sources:
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/pim_config.h`: options.
- `pim.h` / `pim.cc`: ISA, PIM unit, kernel generator.
- `dram_pim.cc`: in-order PIM controller.

## Config

`A100_HBM2PIM` (`SM80_HBM2PIM`) is an A100-class GPU whose DRAM mirrors
PIMSimulator's `ini/HBM2_samsung_2M_16B_x64.ini` and `system_hbm_64ch.ini`:
- 64 pseudo-channels, 16 banks in 4 bank groups, 8 PIM blocks per channel;
- 1 GHz DRAM clock, 32 B per column command;
- Scheme8 address mapping;
- HBM2 timings, including refresh staggered over 2 ranks.

Extras:
- `A100_HBM2PIM-PIM16CH-SASS`: 16 channels.
- `A100_HBM2PIM-NOPIM-SASS`: PIM model off.

## Launching PIM kernels

Add lines to a `kernelslist.g`, between or instead of traced kernels:

```
PIMKernel,gemv,out=4096,in=4096[,batch=1]
PIMKernel,add,n=1048576
PIMKernel,mul,n=2097152
PIMKernel,relu,n=4194304
PIMKernel,stream,baseline=gemv,out=4096,in=4096   # same traffic without PIM
PIMKernel,stream,rd_bytes=<B>,wr_bytes=<B>        # custom read-then-write traffic
```

Optional arguments:
- `channels=N`: number of channels the data is spread over (default: all).
- `sim_channels=N`: simulate only the first N of them. Every channel runs the
  same program, so `sim_channels=1` gives the same cycle count much faster.
- `fix_layout=1`: use `j*batch+b` instead of PIMSimulator's `j+b` for the GEMV
  result column.

How a PIM kernel runs:
- It waits until earlier kernels finish.
- Each channel drains its GPU requests, then runs the generated command stream:
  park-in, SB→HAB, CRF programming, compute, HAB→SB, park-out.
- The channel serves no GPU request until the stream finishes.
- It is reported like a kernel: `kernel_name = PIMKernel_<op>_...`.
- Its summary is printed between the `PIM-kernel-stats` markers:
  `pim_kernel_dram_cycles`, command counts, mode cycles, CRF instructions, MAC
  lane operations, internal bandwidth, and self-checks against
  PIMSimulator's closed-form command counts.

Collect the stats with:

```
./util/job_launching/get_stats.py -k -s ./util/job_launching/stats/pim_stats.yml -N <run>
```

## Options

| Option | Meaning |
|---|---|
| `-gpgpu_pim_enabled` | PIM model on (needs `-gpgpu_simple_dram_model 0`) |
| `-gpgpu_pim_num_blocks`, `-gpgpu_pim_banks_per_block` | PIM blocks per channel, banks per block (2) |
| `-gpgpu_pim_reg_row`, `_abmr_row`, `_sbmr_row`, `_park_row`, `_reserved_row_bit` | reserved rows (PIMSimulator values) |
| `-gpgpu_pim_trans_queue_size`, `_cmd_queue_size`, `_cmd_latency` | PIM controller queues and tCMD |
| `-gpgpu_pim_selfcheck` | print PASS/FAIL against the closed-form counts |
| `-gpgpu_pim_align_refresh` | restart refresh countdowns at PIM kernel launch |
| `-gpgpu_pim_cmd_trace <file>`, `-gpgpu_pim_cmd_trace_ch <n>` | DRAM command trace of one channel |

New `-gpgpu_dram_timing_opt` keys for every config: `RRDL` (tRRD_L), `RCDWR`,
`RTW`, `RTP`, `REFI` and `RFC` (refresh). They default to 0, which keeps the
previous behavior.

`-gpgpu_dram_refresh_ranks` staggers refresh over ranks as DRAMSim2 does.

## Validation against PIMSimulator

```
source gpu-simulator/setup_environment.sh
./util/pim/run_pim_parity.sh -p ~/project/PIMSimulator -t
```

This runs PIMBench's kernels on both simulators and compares cycle counts. With
`-t` it also compares channel 0's DRAM command traces, using
`compare_cmd_trace.py`.

Expected result: every kernel matches exactly. Two command-trace differences
are reporting artifacts of PIMSimulator, and the comparison tool handles both:
- It does not print commands issued in the last cycle of a run.
- It prints PIMRank messages one cycle later than Rank messages.

| PIMBench kernel | PIMSimulator | Accel-Sim |
|---|---|---|
| GEMV 4096x4096, PIM / no PIM | 13166 / 36082 | 13166 / 36082 |
| MUL 2M, PIM / no PIM | 5926 / 13255 | 5926 / 13255 |
| ADD 1M, PIM / no PIM | 3349 / 6651 | 3349 / 6651 |
| RELU 4M, PIM / no PIM | 7665 / 17504 | 7665 / 17504 |

## Model notes and limitations

- **PIMSimulator's scheduling is reproduced as is**:
  - in-order issue with barriers;
  - activates looking past a barrier at the head of the queue;
  - eager precharge;
  - refresh that does not hold back activates;
  - a barrier that is dropped when exactly 64 transactions are queued.
- **All-bank commands are timed on banks 0 and 1 only**, as in PIMSimulator.
  tFAW is not enforced.
- **PIM commands are kept out of the GPU-facing counters.** They are not
  counted in the GPU DRAM stats (`n_rd`, `n_act`, ...) or in AccelWattch. They
  appear in the `PIM[ch]` lines and in the PIM kernel summary.
- **Inherited GEMV quirks are kept and warned about:**
  - a GEMV with a single input tile encodes an odd-bank JUMP count of -1;
  - an element-wise CRF ends on its JUMP, not on EXIT
    (`pim_kernel_pim_off_before_exit`).
- **Not modeled:** PIM kernels concurrent with GPU kernels, GEMV with adder
  tree, batch norm, data values, PIM energy, and PIM issued by SM
  instructions.
