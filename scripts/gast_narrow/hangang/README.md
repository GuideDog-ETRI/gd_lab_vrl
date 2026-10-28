# KISTI Hangang (GH200, aarch64) job scripts for GAST-Narrow v2.3

Site copies of the Slurm scripts used on Hangang natStr (2026-10-28). Paths point at /scratch/t515a05/gd_lab
(SIF, venv, repo, logs); adjust `G=` before reuse. Jobs must be submitted from /scratch or /appsdata.

- `build_env.sbatch`: aarch64 SIF from `containers/gd_lab_isaaclab.def` + host venv (torch pinned by
  `pip-constraints-gh200.txt`, tensordict < 0.15).
- `smoke_v23.sbatch`, `pilot_v23.sbatch`, `scale_v23.sbatch`, `diag_v23.sbatch` (+ `diag_nccl.env`): one-node smoke,
  envs-per-GPU sweep, multi-node scaling and hang diagnostics.
- `train_v23_main.sbatch`: multi-node v2.3 training (`sbatch -N <nodes>`), `switch_to_128.sh`: move a running job to
  more nodes through a saved checkpoint.
- `dist_probe.py`, `probe_*.sbatch`: NCCL-only probes (TCP sockets vs AWS-OFI).
- `nvidia_icd_gh200.json`: Vulkan ICD for the container (`--nv` does not bind the host's `nvidia_icd.aarch64.json`).

Hangang specifics the launcher (`scripts/gast_narrow/train_narrow_v23.sh`) and these scripts handle:
LD_PRELOAD libgomp + libcarb for aarch64 isaacsim; `srun --cpus-per-task=$SLURM_CPUS_ON_NODE`; NCCL over Slingshot
with the host aws-ofi-nccl 1.20.0 + libfabric 2.3.1 (hwloc / libnl-3 / libpciaccess bound from the host);
`NCCL_PROTO=^LL128`; static torchrun rendezvous on a job-specific port; per-rank stdout/stderr files
(`--log_dir ... --redirects 3`, needed beyond ~100 nodes); per-node Kit home on /tmp; Kit console at Error.
