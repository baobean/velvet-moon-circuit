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
image_backup_url`. **Key wins:** `organ` = `part-of` for free — **confirmed vocabulary
from the real (72%-downloaded) CSV, 77,786 rows scanned 2026-09-17: `{leaf, flower,
fruit, bark, habit, branch, scan}`** (7 organ types, not the guessed 5); counts so far
skew toward leaf/flower (~21-22K each), habit ~15.5K, fruit ~9.7K, bark ~7.1K, branch/scan
rare (677/596) — branch and scan may be too sparse for a reliable per-species part crop,
recheck once the full CSV is in. **Gotcha found during this scan:** `csv.DictReader` hit
`field larger than field limit` at row 77,786 — not yet root-caused (could be the
in-progress download's truncated tail, or a genuine unescaped `"` inside a free-text
`author`/`references` field corrupting quote balance downstream). **`build_plantclef.py`
must parse defensively** — catch/skip a malformed row rather than let one bad quote
abort the whole build — and this must be re-verified once the CSV finishes downloading
(currently still running self-healing on the original host, see Infra section). `species/
genus/family` = taxonomy hubs for free; **`url`
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
Extracted 2026-09-17 (11:00, first extraction died silently ~38/200 species in on the
prior host — re-ran to completion, all 200 species / 11,788 images present) at
`/mnt/mmlab2024nas/ldtuan/data/partgraph/cub/CUB_200_2011/`.

**CUB Task 6 reconnaissance (Task 6 Step 1, frozen 2026-09-17) — no per-part bounding
boxes exist**, only point annotations. Layout:
- `images.txt`: `<image_id> <species_dir>/<filename>`; `image_class_labels.txt`:
  `<image_id> <class_id>`; `classes.txt`: `<class_id> <NNN.Species_Name>`.
- `bounding_boxes.txt`: **one whole-bird box per image** — `<image_id> <x> <y> <w> <h>`.
- `parts/parts.txt`: 15 named point parts (`back, beak, belly, breast, crown, forehead,
  left eye, left leg, left wing, nape, right eye, right leg, right wing, tail, throat`).
  `parts/part_locs.txt`: `<image_id> <part_id> <x> <y> <visible>` — a **point**, not a
  box; `(x,y)=(0,0), visible=0` when absent. The plan text ("deriving part crops from
  part_locs.txt bounding boxes") was wrong on this point — there are no per-part boxes,
  only points + the one whole-bird box.
- Top-level `attributes.txt` (312 lines, outside the tgz, fetched separately) names
  attributes as `has_bill_shape::curved_(up_or_down)` etc.; per-image labels are in
  `attributes/image_attribute_labels.txt` (`<image_id> <attribute_id> <is_present>
  <certainty_id> <time>`), per-class in
  `attributes/class_attribute_labels_continuous.txt`.
- Sample image resolution: ~320×223 (varies per image, not fixed).

**PlantCLEF CSV fully downloaded and re-verified 2026-09-17 (later same day)** — 786 MB,
1,408,033 data rows. The `field larger than field limit` error flagged earlier was just
Python's default `csv.field_size_limit()` (131072 bytes), not row corruption —
`csv.field_size_limit(sys.maxsize)` before opening parses the whole file cleanly, no
skipped/malformed rows. Full-file organ counts (supersede the 72%-partial estimate above):
`leaf` 340,852, `flower` 389,251, `fruit` 165,855, `bark` 88,507, `habit` 355,732, `branch`
59,632, `scan` 8,204 — all seven organs healthy, branch/scan no longer look sparse at full
scale. **`build_plantclef.py` must still `csv.field_size_limit(sys.maxsize)`, but the
defensive skip-malformed-row path is now lower priority** (no malformed rows found in the
full file); keep it as a guard, not the primary concern.

**Correction — `url` is NOT always `bs.plantnet.org`.** PlantCLEF2024 aggregates multiple
occurrence platforms (GBIF-linked): observed hosts include `bs.plantnet.org/image/o/<hash>`,
`inaturalist-open-data.s3.amazonaws.com/photos/<id>/original.jpg(eg)`, and
`observation.org/photos/<id>.jpg`. `build_plantclef.py`'s fetch step must do a generic
HTTP GET on the `url` column verbatim (`requests.get(url, timeout=...)` with retry), not
assume a single host or reconstruct the URL from `image_name`.

**Frozen wk2 pilot species list (13, decided 2026-09-17, after wk1 screen)** — drawn from
the wk1 screen's 30-species PlantCLEF sample per
`docs/superpowers/2026-09-17-wk1-rarity-headroom-screen-finding.md`: the 6-species
confusable pocket (leave-one-out retrieval acc. ≤0.60) plus the next-weakest tier
(acc.=0.80), **minus `Marsilea batardae Launert`** — dropped under a new eligibility
criterion frozen here (applied before any FLUX generation or scoring, so it is a screening
criterion like the wk1 gate, not post-hoc cherry-picking): **a species needs ≥2 distinct
`organ` labels in the CSV to test the mechanism at all** — `Marsilea batardae` has only
`habit` (10/10 rows), so its `partgraph` arm would degrade to a single crop, identical in
kind to `single_medoid`, contributing no signal to the GAIN test. No replacement was drawn
(13 is close enough to "~15" and preserves the confusable-pocket provenance; a fresh draw
would mix an unscreened species into a pre-registered list). Per-species organ counts
(full CSV, confirms ≥2 organs and 6-20 images each, this is the ONLY source of images —
do not add extra images beyond what these counts list):

| species | species_id | n_images | organs |
|---|---|---|---|
| Sisymbrium polyceratium L. | 1358432 | 19 | habit 6, branch 9, leaf 1, fruit 1, scan 2 |
| Narcissus viridiflorus Schousb. | 1360973 | 19 | leaf 1, flower 11, branch 4, habit 1, fruit 2 |
| Campanula petraea L. | 1398374 | 10 | leaf 2, flower 2, branch 2, habit 4 |
| Rostraria litorea (All.) Holub | 1361398 | 13 | flower 2, habit 4, fruit 4, bark 1, leaf 1, branch 1 |
| Euphorbia akenocarpa Guss. | 1358523 | 6 | branch 5, habit 1 |
| Astragalus turolensis Pau | 1359157 | 6 | habit 2, flower 4 |
| Taraxacum oblongatum Dahlst. | 1390426 | 9 | habit 4, flower 2, fruit 3 |
| Euphorbia graminifolia Vill. | 1392165 | 15 | leaf 7, flower 2, bark 1, branch 4, habit 1 |
| Agrostis × murbeckii Fouill. | 1647575 | 20 | bark 10, habit 6, branch 4 |
| Desmazeria sicula (Jacq.) Dumort. | 1361240 | 14 | flower 4, fruit 4, leaf 1, habit 4, scan 1 |
| Teucrium turredanum Losa & Rivas Goday | 1564438 | 7 | leaf 1, habit 3, branch 2, flower 1 |
| Thapsia scabra (...) | 1744638 | 20 | leaf 3, habit 9, flower 1, branch 3, fruit 3, bark 1 |
| Hemionitis guanchica (Bolle) Christenh. | 1722699 | 11 | habit 5, leaf 6 |

Match rows by the CSV's `species` column string (exact match against the names above, e.g.
`"Sisymbrium polyceratium L."`) — the `species_id` in the table is for cross-checking, not
matching (a `species_id` maps 1:1 to a `species` string in this CSV, confirmed above). Five
of these species' first 5 images are already staged at
`/mnt/mmlab2024nas/ldtuan/data/partgraph/plantclef2024/wk1_screen_images/<species_id>/` from
the wk1 fidelity check — reuse those files instead of re-fetching (check existence before
GET), but each species needs its FULL image set per the table above (wk1 only staged 5/species),
so most images per species still need fetching.

**Frozen decisions for `build_cub.py`** (crop-derivation rule not specified by the plan —
resolved here so the implementer doesn't invent it independently):
- **Part-type grouping** (15 points → 6 crop types): `head` = {beak, crown, forehead,
  left eye, right eye, throat, nape}; `back` = {back}; `breast` = {breast, belly};
  `wing` = {left wing, right wing}; `leg` = {left leg, right leg}; `tail` = {tail}.
- **Source photo per species**: pick the one image with the most visible part-groups
  (ties broken by lowest `image_id`) as the single "source photo" all of that species'
  part crops are cut from — keeps crops from one consistent specimen/pose, not mixed
  across individuals. The medoid is chosen independently over all eligible images via
  the existing `siglip-centroid-nearest` rule (same as `build_treevill.py`), so medoid
  and part-crop source photo may differ.
- **Crop box**: for a part-group, average the `(x,y)` of its *visible* sub-parts in the
  source photo → crop center. Square crop, `side = max(32, round(0.45 * max(bbox_w,
  bbox_h)))` using that image's whole-bird box from `bounding_boxes.txt`. Clamp by
  *shifting* (never shrinking) the box to stay inside the image; if the image is smaller
  than `side` on an axis, use the full image extent on that axis. A part-group with zero
  visible sub-parts in the source photo is skipped for that species (not fabricated).
- **Medoid-selection helper location correction**: the plan says "reuse the existing
  helper in `build.py`" — it is actually `_centroid_nearest_index` in
  `ragregen/mmkg_store/build_treevill.py` (`build.py` only orchestrates; `build.py` has
  no medoid helper of its own).
- **Encoder**: reuse `ragregen.mmkg_store.embed_index.LOCKED["encoder"]` (same constant
  `build.py.DEFAULT_ENCODER` aliases) for all embeddings — never the DINO eval encoder.

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
