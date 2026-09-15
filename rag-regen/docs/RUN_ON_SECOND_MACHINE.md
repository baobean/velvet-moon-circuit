# Moving the doc-5 campaign to the other machine

Everything that matters — repo, corpus, `index.faiss`, `outputs/`, the HF
weight cache, and `logs/campaign/` — is on the shared NAS at
`/mnt/mmlab2024nas`. So the campaign is *portable*: the second machine picks
up exactly where this one stopped, and no stage is redone.

Written 2026-08-08 from host `mmlab`, where the local RTX 4090 has been held
by a neighbour's job for 19 hours.

## The one rule

**Only one driver may run at a time, on either machine.**

`logs/campaign/` and `outputs/` are shared. Two drivers would both claim the
same stage, both write `index.faiss`, and interleave into one `campaign.log`.

`campaign.sh` now enforces this with a heartbeat file, `logs/campaign/OWNER`,
refreshed every 60s by the running driver. Starting a second one prints
`REFUSING TO START` and exits 9. A claim older than 300s is treated as dead
and taken over, so a crashed machine cannot block the campaign forever.

Check who holds it from either box:

```bash
cat logs/campaign/OWNER          # host pid timestamp
ls -l logs/campaign/OWNER        # mtime = last heartbeat
```

## Step 1 — preflight (2 minutes)

```bash
cd /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen

# The conda env is on the NAS, so it comes along -- but it was built against
# THIS box's driver. Confirm it actually runs there before trusting it.
/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -c \
  "import torch, faiss; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# Should be 512 passed. If the env is wrong, this tells you immediately
# rather than four hours into a GPU stage.
/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q
```

Never use bare `python` — it is a conda 3.13 with no faiss.

## Step 2 — pick the right GPU

This is the part that will silently waste a day if you get it wrong.

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

The pipeline stages ask for **17 GB free**. On a box with 2x16GB + 1x24GB,
**only the 24 GB card can ever satisfy that** — pointed at a 16 GB card,
`gpu_wait` would poll politely forever against hardware that physically
cannot fit the job.

`CAMPAIGN_GPU=auto` picks the card with the most *total* memory, which is the
one you want. Or pass the index by hand if you prefer.

## Step 3 — start it

```bash
cd /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen

./scripts/campaign.sh status              # confirm: 6 done, next = screen

CAMPAIGN_GPU=auto setsid nohup ./scripts/campaign.sh \
  > logs/campaign/nohup.out 2>&1 < /dev/null &

sleep 20 && ./scripts/campaign.sh status  # should show screen RUNNING
tail -f logs/campaign/campaign.log        # heartbeats once a minute
```

`setsid` detaches it: closing the terminal, or the Claude session ending, does
not stop it. The first log lines name the host and the chosen GPU — check
that it picked the 24 GB card.

## What it will do, unattended

`screen` (~1h) -> `dry_run` stitch -> `real_run` inpaint -> `report`.
Each waits for VRAM on the chosen card first, so long silent gaps with a
once-a-minute `MiB free (need ...) -- waiting` line are normal and healthy.

Already done, and skipped on startup: `select`, `download`, `manifest`,
`make_cases`, `validate`, `build_index`. The index is verified — 56,657
vectors x 1152 dims with an aligned `index.faiss.paths.json`, plus a
`*.verified_bak` copy beside it. That is ~40 minutes of encoding you do not
repeat.

## Two things to know

**`screen` is a human gate the campaign does not wait for.** It writes a
labels CSV and asks for a `verdict` per row, but `run_pipeline` reads only the
drafts, never the verdicts — so `dry_run` and `real_run` proceed unlabelled.
The gate's real question is whether **>=30% of drafts are `fail`**; below that
the concept set is too easy and the benchmark measures very little. Label
before you treat the final numbers as meaningful.

**Read the dry run's report before trusting the real run**, and inspect the
retrieved references for the top-ranked `common` cases. Caption matching has
no sense disambiguation, so a wrong word sense retrieves the wrong object and
scores as `harm`.

## If a stage fails

```bash
./scripts/campaign.sh status              # per-stage state
cat logs/campaign/<stage>.failed          # exit code
./scripts/campaign.sh                     # resume; retries that stage
```

**A 139 (SIGSEGV) means read the log above the crash first.** Unloading
torch+faiss+PIL together is unstable in this env. On 2026-08-07 `build_index`
segfaulted *after* writing a complete index and printing its success line, and
the driver then spent an attempt re-encoding 56,657 images for nothing.
`env.exit_now()` now skips teardown in `build_index`, `screen_premise` and
`run_pipeline`, so this should not recur — but if it does, check the artifacts
before letting a retry burn an hour.

Full history, including hazards that already bit, is in
`logs/campaign/RESUME.md`.

## Getting Claude on that machine up to speed

`CLAUDE_CONFIG_DIR` points at the NAS, so both the memory and the transcripts
are already shared:

```bash
export CLAUDE_CONFIG_DIR=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.claude
cd /mnt/mmlab2024nas/ldtuan/code    # must match, it keys the project by cwd
claude --resume                      # pick session 5fb07d65-8353-...
```

`--resume` reopens this exact conversation with full context. Starting fresh
instead still loads `MEMORY.md` and `memory/doc5-benchmark-campaign.md`
automatically, which carry the campaign's state and its hazards — then point
it at this file.
