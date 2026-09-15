# FewMedical-XJAU as external-validation dataset for OracleHub — acquisition scope

**Date:** 2026-09-07
**Purpose:** parallel scoping (per user decision "Treevill now, FewMedical later") of whether/how to
use **FewMedical-XJAU** (Sensors 2025, 25(17), 5499) as a *cross-dataset* test of the OracleHub
taxonomy hypothesis, once the in-dataset Treevill result is in. No GPU; desk research.

## What the dataset is (from the paper)

| property | value | implication for us |
|---|---|---|
| images | 4,992 | small overall |
| species | **540** | huge taxonomic diversity |
| taxonomy | **5 phyla, 125 families**, Phylum–Family–Genus–species, Linnaean | **authoritative built-in family labels — no hand-curation, the oracle signal is provided** ✅ |
| per-species dist. | **long-tail: 80% of species hold 25% of images** (~9/species mean, median much lower) | most species too image-poor to be query concepts ⚠️ |
| organs shown | roots, stems, leaves, flowers, fruits; 3 shot scales (wide/medium/close-up) + whole plant | **part vocabulary ≠ ours** (leaf/bark/branching/cone_or_flower) ⚠️ |
| plant types | rare medicinal plants (many herbs/shrubs, not trees) | **"bark"/"branching" often absent** ⚠️ |
| license | CC BY | reusable once obtained ✅ |
| availability | **no data-availability URL in the paper** | must request from authors / find a mirror ⚠️ |

Sources: [PMC12431608](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431608/) ·
[doi:10.3390/s25175499](https://doi.org/10.3390/s25175499)

## Why it's attractive for OracleHub

1. **The oracle is free and authoritative.** The dataset ships a full Linnaean hierarchy (125
   families). Our Treevill taxonomy is hand-curated best-effort; here family membership is a dataset
   label, removing the "verify the taxonomy" caveat entirely.
2. **Density.** 125 families across 540 species → many multi-member families, so far more Gate-A
   candidate cells than Treevill's 10 families — *if* enough species clear the image-count bar.
3. Independent domain and imagery → genuine external validity for a Treevill-positive result.

## The three real risks (why it is a mini-project, not a drop-in)

1. **Long-tail image scarcity — the same blocker as Treevill, worse.** build_P=0 needs ≥2 unique
   images per (species, part) to form target+scoring. With 80% of species holding 25% of images,
   only the head (~top 20%, ~108 species) is plausibly viable, and even those may give 1–2 held-out
   crops per organ. Net viable-cell count is uncertain until measured.
2. **Part-vocabulary mismatch — a pipeline change, not just data.** Our extractor detects
   leaf/bark/branching/cone_or_flower with GroundingDINO tree-part phrases. Medicinal herbs have no
   "bark" and often no woody "branching"; the dataset's organ vocabulary is root/stem/leaf/flower/
   fruit. To use it we must **remap parts to organs** (drop bark/branching; add stem/flower/fruit) in
   `kg_build.PART_NAMES` / `_part_phrase`, and re-validate that GroundingDINO fires on close-up organ
   shots (some images are whole-plant/wide — detection may be unreliable).
3. **Acquisition friction.** No download link is published. First step is to obtain it (see below);
   timeline depends on author response.

## Concrete acquisition plan (no GPU; do only if Treevill is positive)

1. **Get the data.** In order of preference: (a) search Zenodo / Figshare / PapersWithCode / GitHub
   for "FewMedical-XJAU" mirror; (b) email the corresponding author (Gulimila Kezierbieke, College of
   Computer and Information Engineering, Xinjiang Agricultural University) citing the CC-BY license and
   requesting the release; (c) check the Jiangnan University co-authors' pages. Store under
   `data/fewmedical_xjau/` mirroring the Treevill per-species-dir layout.
2. **CPU precondition FIRST (mirror the Treevill flow).** Before any GPU: parse the provided taxonomy
   into `concept -> family`; using the dataset's own labels, keep only species with ≥2 usable images;
   (this needs organ crops → so a lightweight embed pass is required first, step 3). The gate is the
   same: a same-family sibling whose SigLIP2 rank is > K. Report viable cells/families exactly as
   `reports/2026-09-07-oraclehub-precondition.md`. **If too few viable cells, stop — same rule.**
3. **Pipeline adaptation (GPU).** Remap `PART_NAMES` to organs (leaf, flower, fruit, stem), rerun the
   part-extraction + SigLIP2 exemplar build on the head species, then the precondition, then reuse
   `graft/oraclehub_run.py` unchanged (it is organ-agnostic — it keys on whatever "part" strings the
   store carries) with `graft/taxonomy.py` swapped for the dataset's family table.
4. **Cost estimate:** acquisition (author-response-bound) + ~a few GPU-hours for the head-species
   extraction/embedding + the CPU precondition. Only the restore run itself (small, build_P=0) after.

## Recommendation

Hold FewMedical-XJAU as the **external-validation** step, contingent on a positive Treevill OracleHub
result. Its authoritative 125-family taxonomy makes it the right confirmation dataset, but the
long-tail scarcity and the tree→organ part-vocabulary change mean it is a scoped follow-on, not a
substitute for the Treevill run now in progress. Kick off step 1 (acquisition) in parallel since it is
latency-bound and free; defer steps 2–4 until Treevill reads out.
