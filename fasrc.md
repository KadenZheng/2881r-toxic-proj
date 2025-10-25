# FASRC Guide

This document describes how to get access to the cluster from the command line. Once you have that access you will want to go to the **Running Jobs** page to learn how to interact with the cluster.
- **Do not run jobs or heavy applications (e.g., MATLAB, Mathematica) on the login server.** Use an interactive session or job for all applications and scripts beyond basic terminals, editors, etc. For graphical applications use **Open OnDemand**.
- If you did **not** request cluster access when signing up, you will not be able to log into the cluster or login node (no home directory). You’ll be prompted for a password repeatedly. See the doc on how to add cluster access and additional groups.
- **Shells:** The cluster uses **bash** for the global environment. Using an alternate shell may break things; it’s unsupported. The module system assumes bash.

## Login Nodes

- SSH to `login.rc.fas.harvard.edu` to reach one of the login nodes.
  - Target a specific datacenter:
    - **Boston:** `boslogin.rc.fas.harvard.edu`
    - **Holyoke:** `holylogin.rc.fas.harvard.edu`
  - You can also connect to a specific login node by hostname.
  - **No VPN required.** Accessible worldwide.
- **Purpose:** Login nodes are shared gateways, not for production work. Submit jobs for production work; for interactive work, spawn an interactive job on a compute node. Use Open OnDemand for graphical support.
- **Per-session limits:** **1 core**, **4 GB RAM**, **max 5 sessions per user**. Abuse may result in terminated sessions. Login nodes are rebooted during **monthly maintenance** to clear stale sessions.
- If you need >5 sessions, prefer batch jobs, Open OnDemand (multiple terminals on a dedicated compute node), or tools like **screen**/**tmux**.

## Connecting via SSH

- From macOS/Linux: `ssh USERNAME@login.rc.fas.harvard.edu`
- Enter your password, then the current 6-digit **OpenAuth** token (verification code).
- **Always include your cluster username** in the SSH command to avoid your local username being used.

### SSH Clients

**macOS / Linux / UNIX**
- macOS: **Terminal** (Applications → Utilities) or alternatives like **iTerm2**.
- Linux: use your distro’s terminal; **Tilix** is a popular alternative.

**Windows**
- **Windows Terminal / OpenSSH** (Windows 10+ has ssh built in)
- **WSL** (Windows Subsystem for Linux) for a Linux environment and native tools
- **PuTTY**: enter `login.rc.fas.harvard.edu` as Host Name; supports basic X11 forwarding
- **Git Bash** (Git for Windows)
- **MobaXterm** (free/paid; supports X11 forwarding)
- **XMing** (standalone X11; more complex)

---

# Running Jobs (Slurm)

## Overview

**Slurm** (Simple Linux Utility for Resource Management) is the cluster scheduler. You write a batch script and submit it; Slurm queues and runs it on a partition (queue) you specify. Key features:
- **Stop and Requeue** behavior is robust (e.g., memory-aware requeue).
- **Memory requests** are guaranteed; you cannot exceed what you request.
- **GRES** for fair scheduling of GPUs and other accelerators.
- **Accounting DB** (history, usage) used for job priority/fair-share.

**Work is generally done from the command line.** After account setup, SSH into a login node to begin.

**Do not run applications from login nodes.** Use interactive sessions or the VDI for GUI tools (MATLAB, RStudio, Jupyter).

## Storage for Jobs (Quick Guidance)

- For I/O-heavy workloads, avoid home/lab directories; use `/n/netscratch` (see **Scratch** section).
- **netscratch** is high-performance scratch (VAST), 4 PB total, **50 TB per group**, **90-day retention**, no backups.

## Slurm Resources & Docs

Use:
- `man sbatch` (or other slurm man pages on the cluster)
- Slurm official documentation/tutorials and LLNL guide

## Common Slurm Commands (Summary)

- Submit batch job: `sbatch runscript.sh`
- Interactive run / session (don’t use `salloc` on FASSE):
  - `salloc -p test -t 10 --mem 1000 [script or app]`
  - `salloc -p test -t 10 --mem 1000` (start interactive shell)
- Cancel job: `scancel JOBID`
- View your jobs: `squeue -u USER`
- Check job by ID: `sacct -j JOBID`
- Recurring batch: `scrontab` (see scrontab doc)
- **Job submission caps:** No single user can submit more than **10,000** jobs at a time.

## Slurm Limits

- **Max Jobs per User:** 10,100
- **Max Array Size:** 10,000 (each array index counts toward job total)
- **Max Steps:** 40,000 (each `srun` invocation counts)

## Slurm Partitions (Queues)

> Defaults if not specified: **partition:** `serial_requeue`, **time:** 10 min, **cores:** 1, **memory:** 100 MB.

**Partition overview (selected fields):**
- **sapphire** — 186 nodes, 112 cores/node, Intel *Sapphire Rapids*, **990 GB** RAM, **3 days** max time, **/scratch 396 GB**, **No GPUs**.
  Good for MPI; consider `--contiguous` if topology sensitive (may increase pending time).
- **shared** — 310 nodes, 48 cores/node, Intel *Cascade Lake*, **184 GB** RAM, **3 days**, `/scratch 68 GB`, **No GPUs**.
- **bigmem** — 4 nodes, 112 cores/node, Intel *Sapphire Rapids*, **1988 GB** RAM, **3 days**, `/scratch 396 GB`, **No GPUs**.
- **bigmem_intermediate** — 3 nodes, 64 cores/node, Intel *Ice Lake*, **2000 GB** RAM, **14 days**, `/scratch 396 GB`, **No GPUs**.
- **gpu** — 36 nodes, 64 cores/node, Intel *Ice Lake*, **990 GB** RAM, **3 days**, `/scratch 396 GB`, **GPUs:** 4× A100/node.
- **gpu_h200** — 24 nodes, 112 cores/node, Intel *Sapphire Rapids*, **990 GB** RAM, **3 days**, `/scratch 843 GB`, **GPUs:** 4× H200/node.
- **intermediate** — 12 nodes, 112 cores/node, Intel *Sapphire Rapids*, **990 GB** RAM, **14 days**, `/scratch 396 GB`, **No GPUs**.
- **unrestricted** — 8 nodes, 48 cores/node, Intel *Cascade Lake*, **184 GB** RAM, **no time limit**, `/scratch 68 GB`, **No GPUs**.
- **test** — 18 nodes, 112 cores/node, Intel *Sapphire Rapids*, **990 GB** RAM, **12 hours**, **Max Jobs 5**, **Max Cores 112**, `/scratch 396 GB`.
- **gpu_test** — 14 nodes, 64 cores/node, Intel *Ice Lake*, **487 GB** RAM, **12 hours**, **Max Jobs 2**, **Max Cores 64**, `/scratch 172 GB`, **GPUs:** 8× A100 MIG (3g.20GB) per node (limit 8 per job).
- **remoteviz** — down; 32 cores/node, Intel *Cascade Lake*, **373 GB** RAM, **3 days**, shared V100 GPUs for rendering, `/scratch 396 GB`.
- **serial_requeue** — varies; AMD/Intel; **3 days**, may be GPU-capable, `/scratch varies`.
- **gpu_requeue** — varies; Intel (mixed); **3 days**, GPU-capable, `/scratch varies`.
- **PI/Lab nodes:** varies; no global limits; hardware varies.

Use `spart` to see your partition access. **FASSE** has different partitions than Cannon.

## Submitting Batch Jobs with `sbatch`

Minimal example:

```bash
sbatch runscript.sh
```

Typical submission script:

```bash
#!/bin/bash
#SBATCH -c 1                # Number of cores
#SBATCH -t 0-00:10          # D-HH:MM (min 10 minutes)
#SBATCH -p serial_requeue   # Partition
#SBATCH --mem=100           # Total memory (MB)
#SBATCH -o myoutput_%j.out  # STDOUT file
#SBATCH -e myerrors_%j.err  # STDERR file

module load python/3.10.9-fasrc01
python -c 'print("Hi there.")'
```

**Keep all **`#SBATCH`** lines together at the top.** Slurm copies many environment variables (e.g., `PATH`, CWD). Relative paths are ok.

Key directives:
- `-c N`: core/thread count. Request >1 only if your tool uses them.
- `-t`: time (e.g., `0-01:00`, `minutes:seconds`, etc.). Omitted → 10 minutes default (partition caps still apply). Over-requesting time doesn’t hurt fairshare but can reduce backfill chances.
- `-p`: partition; default is `serial_requeue`.
- `--mem` or `--mem-per-cpu`: **Always specify memory.** Omitted → **100 MB** default (likely too small).
- `-o/-e`: output/error files; `%j` inserts JobID.
- `--test-only`: dry-run to see what would happen.
- `--account=<lab>`: if you’re in multiple labs, charge the right one.

### Email Notifications (use sparingly)

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com
```

Valid `--mail-type` values include: `NONE, BEGIN, END, FAIL, REQUEUE, ALL (avoid), INVALID_DEPEND, STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50, ARRAY_TASKS (avoid for large arrays)`.

**Please avoid **`ALL`** and **`ARRAY_TASKS` on large job sets to prevent email system overload.

### Resource Request Accuracy (esp. memory)

- Accurate `-c` and `--mem`/`--mem-per-cpu` improves efficiency and fairshare outcomes.
- `--mem` is total memory for the job; `--mem-per-cpu` is per core (use for MPI/multinode).
  - Example: `-n 2` with `--mem 4G` ⇒ 2 GB per core; `--mem-per-cpu 4G` ⇒ total 8 GB.
- Use `--test-only` to validate scripts.

## Monitoring & Queues

- `squeue` (live): `squeue -u USER [-l]` for states and details.
- `sacct` (live + historic up to ~6 months):
  - `sacct` → last day’s jobs
  - `sacct -j JOBID` → specific job
  - Use `sstat` for current memory/CPU usage during runtime.
- **Formatting example (array usage):**

  ```
  sacct -j 44375501 --format JobID,Elapsed,ReqMem,MaxRSS,AllocCPUs,TotalCPU,State
  
  ```
- **Summaries:** `seff`, `seff-account`
- **Show broader queue:**
  `showq -o -p shared` (ordered by priority; `-p` selects partition)

**Job states:** `PENDING, RUNNING, COMPLETED, CANCELLED, FAILED`.

## Canceling Jobs

- `scancel JOBID`
- If you lost the JobID, find it via `squeue` or `sacct`.

## Interactive Jobs with `salloc`

> **Note:** On **FASSE**, use the **VDI** instead of `salloc`.

- Example:
  `salloc --partition test --mem 500 --time 0-06:00`
  Starts a shell on a compute node (defaults: 1 core, 100 MB, 10 min if omitted).
- For GUI/X11:
  `salloc --partition test --x11 --mem 4G --time 0-06:00`
- **Don’t add **`/bin/bash` to the `salloc` line; it will run and exit.
- **Idle timeout:** No input for >1 hour terminates the session. For multi-day interactive tasks, print periodically to keep alive.

## Software via Modules

Load required modules (e.g., compilers, Python, MPI, applications) with `module load ...`. See Helmod docs for compiler-dependent “Comp” modules.

## Remote Desktop (VDI)

Use **Open OnDemand** VDI for GUI workflows (MATLAB, RStudio, Jupyter). More reliable than X11 forwarding. Available on **Cannon** and **FASSE**.

## Using GPUs

- Request GPU(s):
  - One GPU: `#SBATCH --gres=gpu`
  - Multiple: `#SBATCH --gres=gpu:N` (per node)
- For heterogeneous partitions (e.g., `gpu_requeue`), constrain GPU class/model:
  - **Constraint by class:** `--constraint="<tag>"`
    - Classes:
      - `nvidia_a100-pcie-40gb`
      - `nvidia_a100-sxm4-40gb`
      - `nvidia_a100-sxm4-80gb`
  - **Exact model:** `--gres=gpu:<model>:1`
    - Example: `--gres=gpu:nvidia_a100-sxm4-80gb:1`
- See available GPUs per partition: `scontrol show partition <PartitionName>` (check **TRES**).

## Parallelization

### Threads / OpenMP

Example:

```bash
#!/bin/bash
#SBATCH -c 8
#SBATCH -t 0-00:30:00
#SBATCH -p sapphire
#SBATCH --mem-per-cpu=100
module load intel/21.2.0-fasrc01
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK MYPROGRAM > output.txt 2> errors.txt
```

### MPI

- Load compiler, then MPI:

  ```bash
  module load intel/21.2.0-fasrc01 openmpi/4.1.1-fasrc01
  
  ```
- Example:

  ```bash
  #!/bin/bash
  #SBATCH -n 128
  #SBATCH -t 10
  #SBATCH -p sapphire
  #SBATCH --mem-per-cpu=100
  module load intel/21.2.0-fasrc01 openmpi/4.1.1-fasrc01
  module load MYPROGRAM
  srun -n $SLURM_NTASKS --mpi=pmix MYPROGRAM > output.txt 2> errors.txt
  
  ```
- Use partitions with MPI interconnects: `sapphire, shared, test, general, unrestricted` (avoid `serial_requeue` for MPI).
- Prefer `--mem-per-cpu` for MPI.
- Consider `--contiguous` if topology-sensitive (may increase pending time).

### Job Arrays

Script (`tophat.sh`):

```bash
#!/bin/bash
#SBATCH -J tophat
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH --mem 4000
#SBATCH -t 0-2:00
#SBATCH -o tophat_%A_%a.out
#SBATCH -e tophat_%A_%a.err
module load tophat/2.0.13-fasrc02
tophat /n/netscratch/informatics_public/ref/ucsc/Mus_musculus/mm10/chromFatrans"${SLURM_ARRAY_TASK_ID}".fq
```

Submit:

```bash
sbatch --array=1-30 tophat.sh
```

- `%A` = JobID, `%a` = array index for filenames.
- `SLURM_ARRAY_TASK_ID` available inside the script.

### Checkpointing

Slurm doesn’t auto-checkpoint. Implement checkpoints in your code (helpful for requeue partitions). Tools: **DMTCP**, **CRIU**.

### Dependencies

Submit dependent jobs:

```bash
sbatch assemble_genome.sh        # returns 53013437
sbatch --dependency=afterok:53013437 annotate_genome.sh
```

- Keep chains short (≤2–3 levels) to avoid scheduler slowdown. Use dependencies when resource needs differ between steps or when waiting on an array’s completion.

### Constraints

Constrain by hardware with `--constraint`. Available feature tags include:
- **CPU:** `amd, intel, avx, avx2, avx512, milan, genoa, skylake, sapphirerapids, cascadelake, icelake`
- **GPU / Network**: see `scontrol show node NODENAME` for specifics.

## Scheduling, Fair-share, and Backfill

- Priority uses **fair-share** (lab-based, 3-day half-life) + **age in queue** (up to 3 days).
  - See priority: `sprio -j JOBID`
  - See fairshare score: `sshare -U`
  - Plan runs to maintain good fairshare.
- **Backfill:** Smaller, well-constrained jobs may run while large jobs wait. Accurate time/memory/CPU requests improve backfill opportunities.

## Troubleshooting

Common errors & causes:
- `CANCELLED ... DUE TO TIME LIMIT`: increase `-t` (e.g., `0-12:30`).
- `exceeded <mem> memory limit, being killed`: increase `--mem`/`--mem-per-cpu` or reduce app usage (e.g., JVM `-Xmx`). For >256 GB, consider **bigmem**.
- `SLURM_receive_msg: Socket timed out on send/recv operation`: scheduler overloaded. Check `sdiag`.
- `CANCELLED ... DUE TO NODE FAILURE`: node lost; jobs are auto-requeued.

---

# Data Transfer

## Copying Data with SCP

**General syntax (FROM first, then TO):**

```
scp [username@server:][from] [to]
```

**From cluster to your machine (run on a FASRC node):**

```bash
scp johnharvard@login.rc.fas.harvard.edu:~/files.zip /home/johnharvard/
# Enter password and OpenAuth PASSCODE
```

**From your machine to the cluster (run locally):**

```bash
scp /home/johnharvard/myfile.zip johnharvard@login.rc.fas.harvard.edu:~/
```

**Recursive copy (entire directory):**

```bash
scp -r johnharvard@login.rc.fas.harvard.edu:~/mydata/ /home/johnharvard/mydata/
```

**Tip:** `~` is your home directory.

## SFTP with FileZilla

Use FileZilla (cross-platform) for SFTP transfers to/from the cluster (and connected resources like shared drives, Dropbox).
**SecureFX note:** disable the “wizard”; under SSH2, enable only **Keyboard Interactive** authentication.

---

# Storage

## Home Directories

- **Path:** `/n/homeNN/USERNAME` (e.g., `/n/home12/jharvard`) or `~`
- **Quota:** **100 GB** (hard), warning at 95 GB
- **Backups:** Daily snapshots retained for 2 weeks, in hidden `~/.snapshot` (not visible in normal `ls`; `cd ~/.snapshot` to access)
- **Availability:** All cluster nodes; can be mounted on desktops/laptops (CIFS/SMB)
- **Retention:** Indefinite
- **Performance:** Moderate; **not** for I/O-intensive or large job counts
- **Security:** Not suitable for HRCI/level 3+ data
- **Tips:**
  - Check usage: `df -h ~`
  - Find large items: `du -h -d 1 ~` or `du -ax . | sort -n -r | head -n 20`
  - If at quota, delete/move files (e.g., to lab or scratch).
  - Over-quota login symptoms: `.Xauthority` lock error (VDI logins fail).

## Lab Directories

- **Path:** `/n/holylabs/<lab>` (as of 2025)
- **Quota:** **4 TiB** (hard), **1 million files**
- **Backups:** Highly redundant, **no backups**
- **Availability:** All cluster nodes; **cannot** be mounted on desktops/laptops
- **Retention:** For duration of lab group
- **Performance:** Moderate; not for heavy I/O or large numbers of jobs
- **Use:** Research data on cluster (not administrative files)
- For more storage options, see Data Storage page.

## Scratch

**Networked, shared **`netscratch` — `/n/netscratch` (or `$SCRATCH`, currently pointing to `/n/netscratch`)
- **Purpose:** High-performance temporary storage for data-intensive computation.
- **System:** VAST parallel file system.
- **Group limits:** **50 TB** per lab, **90-day retention** (purge runs periodically, often during monthly maintenance), **100M inodes** across system, **4 PB total**.
- **Availability:** All cluster nodes; **not** mountable on desktops/laptops.
- **Backups:** None (volatile).
- **Policy:**
  - Intended for transient data; move results you need to retain to lab storage, etc.
  - You may set file times at initial placement to avoid immediate deletion, but **do not** keep touching files to avoid purge—this is abuse and triggers administrative action.
  - If you need longer-term scratch or have concerns, contact FASRC.
- **Use **`$SCRATCH`** variable** in scripts (future-proofing path changes), e.g.:
  `$SCRATCH/jharvard_lab/Lab/jsmith`

**Local (per-node) **`scratch` — `/scratch` (also the backing for `/tmp`)
- **Purpose:** Very fast, **highly volatile** space on the compute node.
- **Size:** Varies by partition (typ. 200–300 GB total per node). See partition table’s `/scratch` column.
- **Scope:** Node-local only (not shared across nodes).
- **Backups/Retention:** None; not retained beyond job duration.
- **Practice:** Clean up at job end; a scratch cleaner runs hourly but don’t rely on it.

---

# Additional Notes & Tips

- **Account setup & access:** If you have an account with VPN and OpenAuth set up, see the Quick Start. If not, see Access & Login.
- **Portals:** You can view your jobs at `portal.rc.fas.harvard.edu/jobs`.
- **Targeting Datacenters:** `boslogin` vs `holylogin` to steer where you land.
- **Username in SSH:** Always specify `USERNAME@...` to avoid local name mismatch.
- **Interactive session keepalive:** Print output periodically for long sessions.
- **Backfill strategy:** Constrain CPU, memory, and time realistically for better fit.
- **Fair-share planning:** Usage is lab-pooled; allow recovery (3-day half-life) before major runs.

---

# Next Steps for kzheng: rsync + run on FASRC (no lab scratch access)

## 1) Test SSH from your Mac

```bash
ssh kzheng@login.fas.rc.harvard.edu
# enter your cluster password, then your 6-digit OpenAuth verification code
exit
```

*(Tip: create an SSH config for convenience—optional)*

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'
Host fasrc
  HostName login.fas.rc.harvard.edu
  User kzheng
  IdentitiesOnly yes
  ServerAliveInterval 60
  ServerAliveCountMax 5
EOF
chmod 600 ~/.ssh/config
# Then you can: ssh fasrc
```

## 2) Prepare a destination folder in your home directory (on the cluster)

```bash
ssh fasrc 'mkdir -p ~/projects/myproject && mkdir -p ~/.venvs'
```

*(We will use your ****home**** directory since you currently don’t have lab scratch access.)*

## 3) Rsync your local Mac codebase to the cluster home directory

Run this **on your Mac** (replace the path to your local project):

```bash
rsync -avh --progress \
  --exclude='.git' --exclude='.DS_Store' --exclude='__pycache__' \
  --exclude='node_modules' --exclude='venv' --exclude='.venv' \
  ~/path/to/local/project/ \
  fasrc:~/projects/myproject/
```

Notes:
- The trailing slashes copy **contents** of the local project into `~/projects/myproject/`.
- Add or remove `--exclude` patterns as needed.

## 4) (Python example) Create/activate a venv and install dependencies

SSH to the cluster and set up your environment in **home**:

```bash
ssh fasrc
module load python/3.10.9-fasrc01
python -m venv ~/.venvs/myproject
source ~/.venvs/myproject/bin/activate
pip install --upgrade pip
# if you have a requirements file:
pip install -r ~/projects/myproject/requirements.txt
```

*(If your project is not Python, replace with your language/toolchain steps using modules as needed.)*

## 5) Quick interactive test (lightweight only)

Use an interactive shell **on a compute node** for brief checks (avoid heavy work on login nodes):

```bash
# small interactive session on the test partition
salloc -p test --mem 1G -t 0-00:30
# once the shell starts on the compute node:
cd ~/projects/myproject
source ~/.venvs/myproject/bin/activate  # if Python
# run a tiny sanity check (quick unit test, --help, etc.)
python -c "print('hello from compute node')"  # example
exit  # exits the compute session when done
```

## 6) Create an sbatch script to run your job on the cluster

Save this as `~/projects/myproject/run_job.sbatch` (adjust resources/commands):

```bash
cat > ~/projects/myproject/run_job.sbatch <<'EOF'
#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH -p serial_requeue
#SBATCH --mem=2G
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
#SBATCH --job-name=myproject_run

module load python/3.10.9-fasrc01
source ~/.venvs/myproject/bin/activate

cd ~/projects/myproject

# >>> Replace the line(s) below with your actual entrypoint <<<
python main.py
# <<< Replace the line(s) above with your actual entrypoint >>>
EOF
```

Submit and monitor:

```bash
cd ~/projects/myproject
sbatch run_job.sbatch
squeue -u kzheng
# after completion:
sacct -j <JOBID> --format JobID,State,Elapsed,ReqMem,MaxRSS,AllocCPUS,TotalCPU
```

Logs will be in `slurm_<JOBID>.out` and `slurm_<JOBID>.err`.

## 7) Update code and re-sync (incremental)

When you make local changes on your Mac:

```bash
rsync -avh --delete --progress \
  --exclude='.git' --exclude='.DS_Store' --exclude='__pycache__' \
  --exclude='node_modules' --exclude='venv' --exclude='.venv' \
  ~/path/to/local/project/ \
  fasrc:~/projects/myproject/
```

*(The* `--delete` *flag removes files on the remote that you deleted locally—omit it if you don’t want that.)*

## 8) Bring results/artifacts back to your Mac (optional)

From your Mac:

```bash
rsync -avh --progress fasrc:~/projects/myproject/outputs/ ~/Downloads/myproject_outputs/
```

## 9) Handy monitoring and troubleshooting commands

```bash
# live jobs
squeue -u kzheng

# recent accounting
sacct --starttime today -u kzheng --format JobID,JobName%20,State,Elapsed,ReqMem,MaxRSS,AllocCPUS,TotalCPU

# job priority breakdown
sprio -u kzheng

# efficiency summary for a finished job
seff <JOBID>
```

Common fixes:
- **Time limit**: increase `#SBATCH -t`.
- **Memory kill**: increase `--mem` or optimize your app.
- **Too heavy for login node**: move the actual run into `sbatch` or an interactive compute session.

HERE ARE THE SCRATCH DIRECTORIES HAVE ACCESS TO: /n/netscratch/kempner_krajan_lab/Lab/kzheng.

AND

/n/holylfs06/LABS/krajan_lab/Lab/kzheng. Prefer the second for anything persistent.

# VEnv Location for Kaden Zheng

**Primary Location:** /n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/
**Symlink:** ~/.venvs → /n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/

**Reason:** Home directory has 100GB quota and was 92% full.
Moved all venvs to lab storage (4TiB quota) to free up space.

**Available venvs:**
- eodnn-cu121
- fishvlm
- gpu-cu121
- manifolds
- olmo-sae
- transcoder-vs-sae

**For safety experiments:** Use conda environment at /n/netscratch/kempner_krajan_lab/Lab/kzheng/conda_envs/align

**Date moved:** October 10, 2025