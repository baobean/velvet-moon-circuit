# Handoff — CVPR 2027 pivot to PartGraph-RAG (keep agents on track)

**Date:** 2026-09-17
**For:** any agent/session picking up this work. Read this, then the spec:
`docs/superpowers/specs/2026-09-17-mmkg-partgraph-generation-design.md`.

## Why the pivot happened (the diagnosis)

The user asked why, after months of complicated MMKG work, nothing "lit up" — while
RAVEL ([2412.09614](https://arxiv.org/abs/2412.09614)) got strong numbers with a
similar-sounding idea. Diagnosis (from reading the whole `docs/superpowers/` trail):

1. **The MMKG kept collapsing to a single medoid image or attribute flags.** Its actual
   structure (parts, relations, multi-reference) was **inert in every experiment**
   (`mmkg_unit_size = 1.0`, `mmkg_part_types = 0.0`). So every arm reduced to a retrieval
   question, and retrieval already wins on single-object identity. The nulls are real and
   correct — but they test the *boundary* of the MMKG hypothesis, not its core.
2. **RAVEL is not "the same method."** RAVEL uses a **textual** KG to shape the
   prompt/conditioning, in the *absence* of visual priors. This project used the graph as
   an image *picker*. Different injection point, different task class → different outcome.
3. **The opening:** nobody has shown the **multimodal** structure (part crops + typed
   relations + multi-reference) helps generation *beyond* one retrieved image. That is the
   CVPR contribution.

## Decisions locked (do not relitigate without the user)

- **Contribution:** a *mechanism* (MMKG → part-structured conditioning) proven on a
  *focused eval*. Not an honest-negative paper.
- **Method (spine):** Approach A — decompose concept via MMKG subgraph → retrieve one
  crop per part → compose on a relation-placed reference canvas → **one** FLUX
  reference-guided pass (no edit-chaining). Reuses the working repair engine
  (`ragregen/regen.py`). B (trained fusion) and C (RAVEL-style text-KG) are **ablation
  arms/baselines, not competing methods.**
- **Datasets:** theme is **biodiversity / fine-grained species broadly** — don't hard-code
  birds+plants. Start: PlantCLEF (primary, rare/hard, native part labels) **+** CUB-200
  (comparability, droppable). **Treevill dropped** — crops too low-res, too few images.
  If a primary fails the wk1 screen, **swap in another biodiversity domain** (iNat
  long-tail insects/reptiles/fungi, Danish Fungi, FathomNet marine), don't abandon the
  direction. Same bar: base model fails on it AND single-image retrieval is weak.
- **MMKG must be enriched** beyond taxonomy (part-of, has-attribute, spatial-arrangement,
  sibling-contrast), **all built from dataset metadata**, or reviewers call it "retrieval
  with a taxonomy."
- **Baselines:** text-only → single-image RAG (IP-Adapter/ImageRAG/single-medoid) →
  text-KG (RAVEL-style) → ours; ablation ladder single-medoid→+parts→+relations→
  +multi-instance.

## Timeline & gates (~7 weeks to CVPR 2027, from 2026-09-17)

- **Wk1:** build MMKG from metadata + **rarity/headroom screen** (kill a dead dataset in
  a day: FLUX must fail on it AND single-image retrieval must be weak).
- **Wk2 GO/NO-GO:** ~15-species pilot — does composed part-structure beat single-medoid
  *and* single-image RAG, CI-clean? Validates FLUX-Kontext multi-part conditioning.
  **If no → pivot mechanism now, don't burn the runway.**
- **Wk3–5** full runs + matrix + ablations. **Wk6** human study. **Wk7** write-up.

## Immediate runnable task (user runs continuously)

The **rarity/headroom screen** — it needs no final mechanism and can start as soon as
PlantCLEF/CUB subsets are staged. Any dataset FLUX already renders well, or where SigLIP
top-1 retrieval is near-ceiling, is dead on arrival (the prior trap).

## Open forks still to confirm with the user

Method name (PartGraph-RAG is a placeholder); whether CUB stays firm or becomes a stretch
goal; "draft + structured repair" as the generation framing. Next process step:
`writing-plans` skill to turn the spec into an implementation plan.

## Prior nulls — do NOT re-run (settled)

MMKG-as-reference-picker and MMKG-as-attribute-gate give no advantage over retrieval on
single-object identity. Triple-confirmed (GRAFT visual repair + rare-attribute transfer;
iNat-birds attribute-verifier; medoid-fidelity pilot). See those findings under
`docs/superpowers/`.

---

## Dataset layouts (confirmed 2026-09-17)

**PlantCLEF 2024** — no login/license needed (CLEF registration optional). Metadata CSV
(`;`-quoted, 750 MB, 1.4M rows) at
`https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv`.
Columns: `image_name;organ;species_id;obs_id;license;partner;author;altitude;latitude;
longitude;gbif_species_id;species;genus;family;dataset;publisher;references;url;learn_tag;
image_backup_url`. **Key wins:** `organ` ∈ {leaf,flower,fruit,bark,habit,...} = `part-of`
for free; `species/genus/family` = taxonomy hubs for free; **`url`
(`bs.plantnet.org/image/o/<hash>`) is a per-image download URL → fetch only our chosen
subset, DO NOT download the 160/281 GB tar.** `learn_tag` = train/test split.
PlantNet's server throttles the tar (and the CSV) to KB/s — use aria2c `-x16` or fetch
per-image from `bs.plantnet.org`. Stage under `/mnt/mmlab2024nas/ldtuan/data/partgraph/`
(4 TB free; NOT `/mmlabworkspace_new`, 690 GB/93%).

**CUB-200-2011** — DOWNLOADED + verified 2026-09-17 →
`/mnt/mmlab2024nas/ldtuan/data/partgraph/cub/CUB_200_2011.tgz` (1.15 GB, 200 species,
11,788 images; has `parts/part_locs.txt`, `parts/parts.txt`, `attributes/`). Gotcha: the
old `vision.caltech.edu/visipedia-data/...` URL is **404**; use the CaltechDATA record
`https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz`, and fetch it with
**`curl -L -C -`** — its presigned-S3 redirect **403s `wget -O`**. Segmentations are a
separate record (`w9d68-gec53`), not needed for Phase 0.

## Infra / GPU notes (runs die often — stay portable and resumable)

**Machines available (run flexibly across all three; GPUs frequently die mid-run):**

| host | note |
|---|---|
| `192.168.6.200` | general runner |
| `192.168.6.202` | general runner — **a 2026-09-15 run threw `CUDA error: unspecified launch failure` mid-pilot** (hardware flakiness; just re-run elsewhere) |
| `192.168.20.245` | has an RTX A5000 (good for FLUX nf4/bf16), **but two gotchas below** |

**`.245` gotchas (cost a day if missed):**
- **Pin the A5000 explicitly:**
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2` — the default `FASTEST_FIRST`
  ordering selects the 12 GB P100 and core-dumps nf4/bf16 FLUX.
- **The queue wrapper's `flock` hangs on `.245`** (NFS lock-manager stall on the
  `192.168.6.133` share). Launch the runner directly, or use a non-flock lock.

**General GPU rules (from the campaign infra):**
- Everything (repo, corpus, `index.faiss`, `outputs/`, HF cache, conda env) is on the
  **shared NAS `/mnt/mmlab2024nas`** → any machine picks up where another stopped; **no
  stage is redone.** Current repo path: `/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/
  velvet-moon-circuit/rag-regen` (older docs say `.../ndbao_hbngoc/rag-regen` — stale).
- **Conda env is on the NAS:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`.
  Never use bare `python` (a conda 3.13 with no faiss). Preflight with a
  `torch.cuda.is_available()` check + `pytest -q` before trusting the env on a new box.
- **One driver at a time across all machines** — `outputs/` and logs are shared. The
  existing heartbeat lock (`logs/campaign/OWNER`, refreshed 60s, dead after 300s) enforces
  it; a second driver prints `REFUSING TO START`. (But note the `.245` flock stall above.)
- **Pick a card with enough free VRAM** (FLUX nf4 ≈ 12 GB; pipeline stages ask ~17 GB
  free). `nvidia-smi --query-gpu=index,name,memory.total --format=csv`. Do not point a job
  at a card that physically cannot fit it — it will poll forever.
- **Make every run resumable and `setsid`-detached** so a dead GPU / closed terminal /
  ended Claude session costs one retried stage, not the whole run. On SIGSEGV (139) when
  unloading torch+faiss+PIL, check whether the artifact was already written before letting
  a retry burn an hour.
- **Claude on another box:** `export CLAUDE_CONFIG_DIR=/mnt/mmlab2024nas/ldtuan/code/
  ndbao_hbngoc/.claude` (memory + transcripts are shared via the NAS).
