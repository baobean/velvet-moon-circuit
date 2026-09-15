# GRAFT Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training-free rare-concept image generator that constructs a visually-grounded multimodal knowledge graph (MMKG) from real reference images and uses it to steer SDXL+IP-Adapter generation with part-level verify-refine, then evaluate it against three baselines on a Treevill subset.

**Architecture:** Three modules behind clean protocols. **M1** reads real refs with Qwen2.5-VL, localizes parts with GroundingDINO+SAM, embeds nodes with SigLIP2/DINOv3, and names a common "anchor" — producing a per-concept MMKG JSON. **M2** composes a KG-structured prompt and conditions SDXL through IP-Adapter on a reranker-selected exemplar. **M3** locates each KG part in the output, scores it against the exemplar, and re-seeds failing generations. An eval harness scores concept fidelity, attribute accuracy, and prompt alignment vs baselines B0/B1/B2.

**Tech Stack:** Python 3, PyTorch 2.6+cu124, diffusers 0.39, transformers 5.14.1, the `kontext` conda env; HuggingFace models already cached under `ndbao_hbngoc/.cache`; pytest with a `gpu` marker.

## Global Constraints

- **Training-free only.** No fine-tuning, LoRA, textual inversion, DreamBooth, or any weight updates. Verbatim from spec §6.
- **Single RTX 4090 (24 GB), models load sequentially, released between stages.** Never hold the VLM and SDXL resident at once. Spec §5.
- **Python interpreter:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Do not create a new env.
- **HF cache:** `HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache`, set via `graft.env.setup()` **before** importing `transformers`/`diffusers`/`huggingface_hub`.
- **Cached model IDs (do not substitute):** SDXL `stabilityai/stable-diffusion-xl-base-1.0`; IP-Adapter `h94/IP-Adapter` (`sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors`, image encoder `models/image_encoder`); VLM `Qwen/Qwen2.5-VL-7B-Instruct`; reranker `Qwen/Qwen3-VL-Reranker-2B`; SigLIP2 `google/siglip2-base-patch16-384`; DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m`; CLIP `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`; GroundingDINO `IDEA-Research/grounding-dino-base`; SAM `facebook/sam-vit-huge`.
- **No sci-fi levers** (spec §6): no cross-attention/regional-attention surgery, no Frankenstein part compositing in the core path.
- **Package root:** `kg_test/graft/`. **Tests:** `kg_test/tests/`. Run pytest from `kg_test/`.
- **Test markers:** pure-logic tests run by default; any test that loads a model or touches CUDA is marked `@pytest.mark.gpu` and deselected by default (`addopts = -m "not gpu"`), mirroring `rag-regen/pytest.ini`.

---

## File Structure

```
kg_test/
  pytest.ini                     # gpu marker, addopts -m "not gpu"
  graft/
    __init__.py
    env.py                       # HF cache + CUDA env setup, free_gpu()
    config.py                    # GraftConfig dataclass (paths, ids, thresholds)
    schema.py                    # MMKG dataclasses + JSON (de)serialize
    interfaces.py                # Protocols: VLM, Reranker, Embedder, Detector, Segmenter, Generator
    models.py                    # concrete loaders implementing the protocols
    dataset.py                   # Treevill loader + build/heldout split
    kg_build.py                  # M1: build MMKG from refs
    prompt.py                    # KG -> structured prompt string
    generate.py                  # M2 Lever A: SDXL+IP-Adapter
    verify.py                    # M3: part localization + scoring + checklist
    refine.py                    # M3: re-seed loop
    pipeline.py                  # concept -> final image (M1->M2->M3)
    metrics.py                   # fidelity/attribute/alignment scoring
    baselines.py                 # B0/B1/B2
    run_eval.py                  # driver over subset -> results.json + table
  tests/
    fixtures/                    # tiny 2-species image fixture + a sample MMKG json
    test_schema.py
    test_dataset.py
    test_prompt.py
    test_metrics.py
    test_selection.py            # reranker/embedder-driven exemplar pick (fakes)
    test_kg_build_gpu.py         # gpu
    test_generate_gpu.py         # gpu
    test_verify_gpu.py           # gpu
    test_pipeline_gpu.py         # gpu
  configs/pipeline.yaml
  data/                          # treevill_archive.zip -> extracted here
  outputs/                       # generated images + MMKG jsons + eval results
```

**Testability principle:** `kg_build`, `generate`, `verify` receive their models as protocol objects (dependency injection). Pure logic is unit-tested with in-memory fakes; the real loaders are exercised only in `gpu`-marked tests. This lets Tasks 6–9 be TDD'd without a GPU.

---

### Task 1: Project scaffold, env, config

**Files:**
- Create: `kg_test/graft/__init__.py`, `kg_test/graft/env.py`, `kg_test/graft/config.py`, `kg_test/pytest.ini`, `kg_test/configs/pipeline.yaml`
- Test: `kg_test/tests/test_config.py`

**Interfaces:**
- Produces: `graft.env.setup() -> None`; `graft.env.free_gpu() -> None`; `graft.config.GraftConfig` (frozen dataclass) with fields `hf_cache: str`, `device: str`, model-id strings (see Global Constraints), `ip_scale: float = 0.6`, `n_seeds: int = 4`, `n_refine: int = 2`, `part_sim_threshold: float = 0.5`, `attr_pass_threshold: float = 0.6`, `k_build_refs: int = 5`, `outputs_dir: str`; `GraftConfig.from_yaml(path) -> GraftConfig`.

- [ ] **Step 1: Write `graft/env.py`** — copy the idiom from `rag-regen/ragregen/env.py`:

```python
import gc, os
from pathlib import Path
HF_CACHE = Path("/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache")

def setup() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
```

- [ ] **Step 2: Write `graft/config.py`** — a frozen dataclass holding every value from Global Constraints plus the tunables above, and a `from_yaml` classmethod (use `yaml.safe_load`; unspecified keys fall back to defaults). Write `configs/pipeline.yaml` with the default values spelled out.

- [ ] **Step 3: Write `pytest.ini`**:

```ini
[pytest]
testpaths = tests
markers =
    gpu: requires the shared RTX 4090 (deselected by default)
addopts = -m "not gpu"
```

- [ ] **Step 4: Write the failing test** `tests/test_config.py`:

```python
from graft.config import GraftConfig

def test_defaults_and_yaml_roundtrip(tmp_path):
    cfg = GraftConfig()
    assert cfg.sdxl_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert cfg.vlm_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert 0.0 < cfg.ip_scale <= 1.0
    p = tmp_path / "c.yaml"
    p.write_text("ip_scale: 0.4\nn_seeds: 2\n")
    cfg2 = GraftConfig.from_yaml(str(p))
    assert cfg2.ip_scale == 0.4 and cfg2.n_seeds == 2
    assert cfg2.vlm_id == cfg.vlm_id  # unspecified keys keep defaults
```

- [ ] **Step 5: Run** `cd kg_test && /mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_config.py -v` — expect FAIL, then PASS after Steps 1–2.
- [ ] **Step 6: Commit** `feat(graft): scaffold, env cache setup, config`.

---

### Task 2: MMKG schema + JSON serialization

**Files:**
- Create: `kg_test/graft/schema.py`
- Test: `kg_test/tests/test_schema.py`

**Interfaces:**
- Produces:
  - `AttributeNode(name: str, value: str, source: str)` — `source` ∈ `{"vision","llm"}`.
  - `PartNode(name: str, attributes: list[AttributeNode], exemplar_crop: str|None, embeddings: dict[str,list[float]])` — `embeddings` keyed by encoder slug (`"siglip2"`,`"dino"`).
  - `ConceptKG(concept: str, global_attrs: list[AttributeNode], parts: list[PartNode], concept_embeddings: dict[str,list[float]], anchor: str, delta: list[str], ref_paths: list[str])`.
  - `ConceptKG.to_json(path)`, `ConceptKG.from_json(path) -> ConceptKG`, `ConceptKG.attribute_texts() -> list[str]` (flatten global + part attrs into `"<part> <name>: <value>"` phrases, vision-sourced first).

- [ ] **Step 1: Write failing test** `tests/test_schema.py`:

```python
from graft.schema import ConceptKG, PartNode, AttributeNode

def _kg():
    return ConceptKG(
        concept="monkey puzzle tree",
        global_attrs=[AttributeNode("silhouette","symmetric candelabra","vision")],
        parts=[PartNode("leaf",[AttributeNode("shape","stiff triangular scales","vision")],
                        "crops/leaf.png",{"siglip2":[0.1,0.2],"dino":[0.3]})],
        concept_embeddings={"siglip2":[0.5,0.5],"dino":[0.9]},
        anchor="conifer", delta=["leaves are stiff triangular scales"],
        ref_paths=["refs/1.jpg"])

def test_roundtrip(tmp_path):
    kg = _kg(); p = tmp_path/"kg.json"; kg.to_json(str(p))
    kg2 = ConceptKG.from_json(str(p))
    assert kg2.concept == "monkey puzzle tree"
    assert kg2.parts[0].attributes[0].value == "stiff triangular scales"
    assert kg2.anchor == "conifer"
    assert kg2.parts[0].embeddings["siglip2"] == [0.1,0.2]

def test_attribute_texts_orders_vision_first():
    kg = _kg()
    kg.global_attrs.append(AttributeNode("age","old","llm"))
    texts = kg.attribute_texts()
    assert texts[0].endswith("symmetric candelabra")   # vision before llm
    assert any("leaf" in t and "triangular" in t for t in texts)
```

- [ ] **Step 2: Run** the test → FAIL.
- [ ] **Step 3: Implement `schema.py`** using `@dataclass` and `dataclasses.asdict`/manual nested reconstruction in `from_json` (json only stores primitives/lists/dicts). `attribute_texts` sorts by `source=="vision"` first, formatting global attrs as `"<name>: <value>"` and part attrs as `"<part> <name>: <value>"`.
- [ ] **Step 4: Run** the test → PASS.
- [ ] **Step 5: Commit** `feat(graft): MMKG schema and JSON serialization`.

---

### Task 3: Treevill dataset loader + split

**Files:**
- Create: `kg_test/graft/dataset.py`, `kg_test/tests/fixtures/` (2 tiny fake species dirs, 3 images each)
- Test: `kg_test/tests/test_dataset.py`

**Interfaces:**
- Produces: `Species(name: str, images: list[str])`; `load_treevill(root: str) -> list[Species]` (each immediate subdir with ≥1 image is a species; recognizes `.jpg/.jpeg/.png/.webp`); `split_refs(sp: Species, k_build: int, seed: int) -> tuple[list[str], list[str]]` returning `(build_refs, heldout_refs)`, disjoint, deterministic by seed.

- [ ] **Step 1: Create the fixture** — write `tests/fixtures/mini_treevill/Akashmoni/{1,2,3}.jpg` and `.../Debdaru/{1,2,3}.jpg` as 32×32 solid-color PNGs saved with `.jpg` extension via PIL (content irrelevant to this task's logic).
- [ ] **Step 2: Write failing test** `tests/test_dataset.py`:

```python
from graft.dataset import load_treevill, split_refs

FIX = "tests/fixtures/mini_treevill"

def test_load_finds_species():
    sp = {s.name: s for s in load_treevill(FIX)}
    assert set(sp) == {"Akashmoni","Debdaru"}
    assert len(sp["Akashmoni"].images) == 3

def test_split_is_disjoint_and_deterministic():
    sp = load_treevill(FIX)[0]
    b1,h1 = split_refs(sp, k_build=2, seed=0)
    b2,h2 = split_refs(sp, k_build=2, seed=0)
    assert b1==b2 and h1==h2                 # deterministic
    assert set(b1).isdisjoint(h1)            # disjoint
    assert len(b1)==2 and len(h1)==1
```

- [ ] **Step 3: Run** → FAIL. **Step 4: Implement** `dataset.py` (`os.scandir`, sorted for determinism, `random.Random(seed).sample`). **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `feat(graft): Treevill loader with deterministic build/heldout split`.

---

### Task 4: Model protocols + concrete loaders

**Files:**
- Create: `kg_test/graft/interfaces.py`, `kg_test/graft/models.py`
- Test: `kg_test/tests/test_verify_gpu.py::test_loaders_smoke` (gpu)

**Interfaces:**
- `interfaces.py` — `typing.Protocol`s:
  - `VLM.describe(images: list[Image], instruction: str) -> str` and `VLM.ask(image: Image, question: str) -> str`.
  - `Reranker.rank(query: str, images: list[Image]) -> list[float]` (higher = more relevant).
  - `Embedder.embed_image(images: list[Image]) -> np.ndarray` and `Embedder.embed_text(texts: list[str]) -> np.ndarray` — **L2-normalized** rows.
  - `Detector.detect(image: Image, phrase: str) -> list[tuple[float,float,float,float]]` (xyxy, pixel coords).
  - `Segmenter.mask(image: Image, box) -> np.ndarray` (bool HxW).
  - `Generator.generate(prompt: str, ip_image: Image|None, seed: int, negative: str, steps: int) -> Image`.
- `models.py` — concrete classes implementing them (`QwenVLM`, `QwenReranker`, `SiglipEmbedder`, `DinoEmbedder`, `ClipEmbedder`, `GroundingDinoDetector`, `SamSegmenter`, `SdxlIpGenerator`) plus a `Models` façade with **lazy** singletons and `unload(name)`.

- [ ] **Step 1:** Write `interfaces.py` with the Protocols above (bodies `...`).
- [ ] **Step 2: Implement `models.py`.** Reuse in-workspace idioms verbatim:
  - `QwenVLM` — mirror `rag-regen/ragregen/vlm.py`: `AutoProcessor` + `Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.vlm_id, torch_dtype=torch.bfloat16, device_map=cfg.device)`; build messages `[{"role":"user","content":[{"type":"image","image":img},...,{"type":"text","text":instruction}]}]`, `processor.apply_chat_template(...)`, generate, decode.
  - `SdxlIpGenerator` — mirror `ImageRAG/imageRAG_SDXL.py:53-80`:
    ```python
    from diffusers import AutoPipelineForText2Image
    from transformers import CLIPVisionModelWithProjection
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",
            subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = AutoPipelineForText2Image.from_pretrained(cfg.sdxl_id,
            image_encoder=enc, torch_dtype=torch.float16).to(cfg.device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    ```
    In `generate`: if `ip_image is None`, call `pipe.set_ip_adapter_scale(0.0)` (or a text-only pipe) else `pipe.set_ip_adapter_scale(cfg.ip_scale)` and pass `ip_adapter_image=ip_image`; seed via `torch.Generator(cfg.device).manual_seed(seed)`; default `negative="monochrome, lowres, bad anatomy, worst quality, low quality"`, `steps=50`.
  - `GroundingDinoDetector`/`SamSegmenter`/`SiglipEmbedder` — mirror `rag-regen/ragregen/models.py` (`GroundingDinoForObjectDetection`, `SamModel`, `AutoModel` for SigLIP). `DinoEmbedder` uses `AutoModel.from_pretrained(cfg.dino_id)` CLS/pooled output; `ClipEmbedder` uses `open_clip` with the cached laion CLIP for `embed_text`/`embed_image`. All `embed_*` return `x / np.linalg.norm(x, axis=1, keepdims=True)`.
  - `QwenReranker` — mirror `rag-regen/scripts/qwen_reranker_score.py` (`MODEL_ID="Qwen/Qwen3-VL-Reranker-2B"`); `rank` returns one relevance score per image for the query.
  - `Models` façade: `models.vlm`, `.reranker`, `.siglip`, `.dino`, `.clip`, `.detector`, `.segmenter`, `.generator` each lazy-built on first access; `unload(name)` deletes and calls `env.free_gpu()`.
- [ ] **Step 3: Write the gpu smoke test** `tests/test_verify_gpu.py`:

```python
import pytest, numpy as np
from PIL import Image
pytestmark = pytest.mark.gpu

def test_loaders_smoke():
    from graft import env; env.setup()
    from graft.config import GraftConfig
    from graft.models import Models
    m = Models(GraftConfig())
    img = Image.new("RGB",(384,384),(120,160,90))
    e = m.siglip.embed_image([img]); assert e.shape[0]==1
    assert abs(float(np.linalg.norm(e[0]))-1.0) < 1e-3     # normalized
    boxes = m.detector.detect(img, "leaf"); assert isinstance(boxes, list)
    m.unload("siglip")
```

- [ ] **Step 4: Run** `... -m pytest tests/test_verify_gpu.py::test_loaders_smoke -v -m gpu` — expect PASS (this is the first real model load; budget a few minutes). If a specific loader OOMs, add `enable_model_cpu_offload()` for that model and note it.
- [ ] **Step 5: Commit** `feat(graft): model protocols and concrete loaders`.

---

### Task 5: M1 — build the visually-grounded MMKG

**Files:**
- Create: `kg_test/graft/kg_build.py`
- Test: `kg_test/tests/test_kg_build_unit.py` (fakes, no gpu) and `tests/test_kg_build_gpu.py` (gpu)

**Interfaces:**
- Consumes: `VLM`, `Detector`, `Segmenter`, `Embedder` (siglip+dino), `Reranker`; `ConceptKG`/`PartNode`/`AttributeNode`; `GraftConfig`.
- Produces: `PART_NAMES = ["leaf","bark","cone_or_flower","branching"]`; `build_kg(concept, build_refs, models, cfg) -> ConceptKG`; helper `parse_vlm_schema(raw: str) -> tuple[list[AttributeNode], dict[str,list[AttributeNode]]]` (pure — parses the VLM's JSON reply into global + per-part attribute nodes, `source="vision"`).

- [ ] **Step 1: Write the failing PURE test** `tests/test_kg_build_unit.py` for `parse_vlm_schema`:

```python
from graft.kg_build import parse_vlm_schema

RAW = '{"global":{"silhouette":"symmetric candelabra"},' \
      '"parts":{"leaf":{"shape":"stiff triangular scales"},"bark":{"texture":"rough"}}}'

def test_parse_builds_vision_nodes():
    g, parts = parse_vlm_schema(RAW)
    assert g[0].name=="silhouette" and g[0].source=="vision"
    assert parts["leaf"][0].value=="stiff triangular scales"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement `parse_vlm_schema`** (robust JSON extraction: find first `{`…last `}`, `json.loads`, tolerate missing parts). **Step 4: Run** → PASS.
- [ ] **Step 5: Implement `build_kg`** (the gpu path). Sequence, releasing each model before the next stage:
  1. `raw = models.vlm.describe(build_refs_as_images, INSTRUCTION)` where `INSTRUCTION` asks for exactly the JSON schema `{global:{...}, parts:{leaf,bark,cone_or_flower,branching}}`, "describe only what is visible".
  2. `anchor_raw = models.vlm.describe(images, ANCHOR_INSTRUCTION)` → `{"anchor": "...", "delta": ["...","..."]}`.
  3. `global_attrs, part_attrs = parse_vlm_schema(raw)`.
  4. For each part in `PART_NAMES`: `boxes = detector.detect(ref0, part_phrase)`; if boxes, `mask = segmenter.mask(ref0, boxes[0])`, crop bbox, save to `outputs/<concept>/crops/<part>.png`; else `exemplar_crop=None`. Compute `siglip`/`dino` embeddings of the crop (or of whole ref0 when no crop) → `PartNode.embeddings`.
  5. `concept_embeddings` = mean of siglip/dino embeddings over all `build_refs` (renormalized).
  6. Use `reranker.rank(concept, build_refs_images)` to store the best ref first in `ref_paths` (drives exemplar choice in M2).
  7. Assemble and return `ConceptKG`; also `to_json(outputs/<concept>/kg.json)`.
- [ ] **Step 6: Write gpu integration test** `tests/test_kg_build_gpu.py`: build on the 2-species fixture (or a real Treevill species once downloaded), assert the JSON exists, has 4 part nodes, non-empty `global_attrs`, a non-empty `anchor`, and each part has a `siglip2` embedding vector.
- [ ] **Step 7: Run** the unit test (default) and the gpu test (`-m gpu`) → PASS.
- [ ] **Step 8: Commit** `feat(graft): M1 visually-grounded MMKG construction`.

---

### Task 6: KG → structured prompt

**Files:**
- Create: `kg_test/graft/prompt.py`
- Test: `kg_test/tests/test_prompt.py`

**Interfaces:**
- Consumes: `ConceptKG`.
- Produces: `compose_prompt(kg, max_attrs: int = 8) -> str`; `compose_ravel_prompt(concept, llm_attrs: list[str]) -> str` (the B2 baseline form — same template, LLM attributes).

- [ ] **Step 1: Write failing test** `tests/test_prompt.py`:

```python
from graft.prompt import compose_prompt
from graft.schema import ConceptKG, PartNode, AttributeNode

def test_prompt_mentions_concept_and_vision_attrs_first():
    kg = ConceptKG("monkey puzzle tree",
        [AttributeNode("silhouette","symmetric candelabra","vision")],
        [PartNode("leaf",[AttributeNode("shape","stiff triangular scales","vision")],None,{})],
        {}, "conifer", [], ["r.jpg"])
    p = compose_prompt(kg)
    assert p.lower().startswith("a photo of a monkey puzzle tree")
    assert "stiff triangular scales" in p
    assert "symmetric candelabra" in p
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `compose_prompt`: `"a photo of a {concept}, " + ", ".join(kg.attribute_texts()[:max_attrs])`, vision attrs first (already ordered by `attribute_texts`). `compose_ravel_prompt` uses the same joining over a plain attribute list. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(graft): KG-structured prompt composition`.

---

### Task 7: M2 — SDXL+IP-Adapter generation + exemplar selection

**Files:**
- Create: `kg_test/graft/generate.py`
- Test: `kg_test/tests/test_selection.py` (fakes) and `tests/test_generate_gpu.py` (gpu)

**Interfaces:**
- Consumes: `Generator`, `Reranker`, `ConceptKG`, `compose_prompt`, `GraftConfig`.
- Produces: `select_exemplar(kg, candidate_paths, reranker) -> str` (pure w.r.t. an injected reranker); `generate_lever_a(kg, models, cfg, seed) -> tuple[Image, str]` returning `(image, prompt)`.

- [ ] **Step 1: Write failing pure test** `tests/test_selection.py` with a fake reranker:

```python
from graft.generate import select_exemplar
from graft.schema import ConceptKG

class FakeReranker:
    def rank(self, query, images): return [0.1, 0.9, 0.3]  # middle wins

def test_select_exemplar_picks_top_ranked():
    kg = ConceptKG("x",[],[],{}, "y", [], ["a.jpg","b.jpg","c.jpg"])
    # patch loader so images don't need to exist -> inject paths directly
    chosen = select_exemplar(kg, ["a.jpg","b.jpg","c.jpg"], FakeReranker(),
                             _open=lambda p: p)
    assert chosen == "b.jpg"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `select_exemplar(kg, candidate_paths, reranker, _open=Image.open)`: open each candidate, `scores = reranker.rank(kg.concept, imgs)`, return `candidate_paths[argmax]`. **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** `generate_lever_a`: `prompt = compose_prompt(kg)`; `exemplar = select_exemplar(kg, kg.ref_paths, models.reranker)`; `img = models.generator.generate(prompt, ip_image=Image.open(exemplar), seed=seed, negative=..., steps=50)`; return `(img, prompt)`.
- [ ] **Step 6: Write gpu test** `tests/test_generate_gpu.py`: load a fixture `ConceptKG` (`tests/fixtures/sample_kg.json` written in Task 5) with a real ref path, run `generate_lever_a`, assert an RGB `PIL.Image` of size ≥ (768,768) is returned and saved.
- [ ] **Step 7: Run** unit (default) + gpu (`-m gpu`) → PASS. **Step 8: Commit** `feat(graft): M2 SDXL+IP-Adapter generation with reranked exemplar`.

---

### Task 8: M3 — part-grounded verification

**Files:**
- Create: `kg_test/graft/verify.py`
- Test: `kg_test/tests/test_verify_unit.py` (fakes) and extend `tests/test_verify_gpu.py`

**Interfaces:**
- Consumes: `VLM`, `Detector`, `Embedder`, `ConceptKG`, `GraftConfig`.
- Produces: `PartScore(part: str, sim: float, present: bool)`; `VerifyReport(part_scores: list[PartScore], attr_pass: float, failing_parts: list[str], ok: bool)`; `verify(image, kg, models, cfg) -> VerifyReport`; pure helper `aggregate(part_scores, attr_pass, sim_thresh, attr_thresh) -> VerifyReport`.

- [ ] **Step 1: Write failing pure test** `tests/test_verify_unit.py` for `aggregate`:

```python
from graft.verify import aggregate, PartScore

def test_aggregate_flags_failing_parts_and_overall():
    ps = [PartScore("leaf",0.7,True), PartScore("bark",0.3,False)]
    r = aggregate(ps, attr_pass=0.8, sim_thresh=0.5, attr_thresh=0.6)
    assert r.failing_parts == ["bark"]
    assert r.ok is False          # a part failed
    r2 = aggregate([PartScore("leaf",0.7,True)], 0.9, 0.5, 0.6)
    assert r2.ok is True
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `aggregate`: `failing = [p.part for p in part_scores if p.sim < sim_thresh]`; `ok = (len(failing)==0) and (attr_pass >= attr_thresh)`. **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** `verify` (gpu path): for each part, `boxes = detector.detect(image, part_phrase)`; if found, crop, embed with siglip+dino, cosine-sim vs the part's stored embeddings (average the two encoders), `present=sim>=threshold`; for attributes, ask `models.vlm.ask(image, "Does this image show <attr value>? Answer yes or no.")` over `kg.attribute_texts()`, `attr_pass = mean(yes)`. Return `aggregate(...)`.
- [ ] **Step 6: Extend** `tests/test_verify_gpu.py` with `test_verify_returns_report`: verify a generated fixture image against `sample_kg.json`, assert a `VerifyReport` with one `PartScore` per part and `0<=attr_pass<=1`.
- [ ] **Step 7: Run** unit + gpu → PASS. **Step 8: Commit** `feat(graft): M3 part-grounded verification`.

---

### Task 9: M3 refine loop + end-to-end pipeline

**Files:**
- Create: `kg_test/graft/refine.py`, `kg_test/graft/pipeline.py`
- Test: `kg_test/tests/test_refine_unit.py` (fakes) and `tests/test_pipeline_gpu.py` (gpu)

**Interfaces:**
- Consumes: everything above.
- Produces: `refine(kg, models, cfg) -> tuple[Image, VerifyReport, str]` (best of up to `1+n_refine` re-seeds by report quality); `run_concept(concept, build_refs, models, cfg) -> dict` (builds KG if absent, generates+refines, saves image + report json to `outputs/<concept>/`).

- [ ] **Step 1: Write failing pure test** `tests/test_refine_unit.py` — inject a fake generator+verifier via a seam so the loop logic is testable without a GPU:

```python
from graft.refine import pick_best

def test_pick_best_prefers_ok_then_higher_score():
    # candidates: (image_tag, report_ok, score)
    cands = [("a", False, 0.4), ("b", True, 0.5), ("c", True, 0.9)]
    assert pick_best(cands) == "c"
    cands2 = [("a", False, 0.8), ("b", False, 0.6)]
    assert pick_best(cands2) == "a"   # none ok -> highest score
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `pick_best` (sort by `(ok, score)` desc) and `refine` (loop seeds `0..n_refine`, generate → verify → collect `(image, report, score=mean part sim * 0.5 + attr_pass * 0.5)`, stop early if `report.ok`, else return `pick_best`). **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** `pipeline.run_concept` tying M1→M2→M3, writing `outputs/<concept>/final.png` and `report.json`.
- [ ] **Step 6: Write gpu test** `tests/test_pipeline_gpu.py`: run `run_concept` on one real Treevill species (small, `n_refine=1`), assert `final.png` and `report.json` are produced.
- [ ] **Step 7: Run** unit + gpu → PASS. **Step 8: Commit** `feat(graft): refine loop and end-to-end pipeline`.

---

### Task 10: Metrics + baselines

**Files:**
- Create: `kg_test/graft/metrics.py`, `kg_test/graft/baselines.py`
- Test: `kg_test/tests/test_metrics.py`

**Interfaces:**
- Produces (metrics): `image_fidelity(gen: Image, refs: list[Image], embedder) -> float` (mean cosine sim, embeddings L2-normalized); `clip_t(gen, text, clip) -> float`; `attribute_accuracy(gen, attr_texts, vlm) -> float`; pure `mean_cosine(gen_vec, ref_mat) -> float`.
- Produces (baselines): `b0_vanilla(concept, models, cfg, seed) -> Image`; `b1_imagerag(concept, pool_paths, models, cfg, seed) -> Image` (SigLIP text→image top-1 from pool + IP-Adapter, mirroring `ImageRAG/retrieval.py`); `b2_ravel(concept, models, cfg, seed) -> Image` (VLM lists attributes from parametric memory → `compose_ravel_prompt` → SDXL text-only, no IP-Adapter).

- [ ] **Step 1: Write failing test** `tests/test_metrics.py` for the pure math:

```python
import numpy as np
from graft.metrics import mean_cosine

def test_mean_cosine_normalized_vectors():
    gen = np.array([1.0,0.0])
    refs = np.array([[1.0,0.0],[0.0,1.0]])   # sims 1.0 and 0.0
    assert abs(mean_cosine(gen, refs) - 0.5) < 1e-6
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `mean_cosine` (assumes normalized rows: `float((refs @ gen).mean())`) and the model-backed metric wrappers (which normalize then delegate). **Step 4: Run** → PASS.
- [ ] **Step 5: Implement `baselines.py`.** `b1_imagerag` reuses the SigLIP text-image retrieval idiom from `ImageRAG/retrieval.py:get_siglip_similarities` (top-1 path) then IP-Adapter-conditions SDXL; `b2_ravel` prompts the VLM "list 8 visual attributes of a {concept}" (no image) → prompt-only SDXL.
- [ ] **Step 6: Commit** `feat(graft): fidelity/attribute/alignment metrics and B0/B1/B2 baselines`.

---

### Task 11: Evaluation driver + first run

**Files:**
- Create: `kg_test/graft/run_eval.py`
- Test: `kg_test/tests/test_eval_unit.py` (aggregation, fakes)

**Interfaces:**
- Consumes: `dataset`, `pipeline`, `baselines`, `metrics`, `GraftConfig`.
- Produces: `evaluate(species_list, methods, models, cfg) -> dict` (per method: mean DINO/SigLIP2 fidelity, CLIP-I, attribute accuracy, CLIP-T over species) writing `outputs/eval/results.json` and a markdown table; CLI `python -m graft.run_eval --root data/... --n-species 12 --methods ours,b0,b1,b2`.

- [ ] **Step 1: Write failing pure test** `tests/test_eval_unit.py`:

```python
from graft.run_eval import summarize

def test_summarize_means_per_method():
    rows = [{"method":"ours","dino":0.8},{"method":"ours","dino":0.6},
            {"method":"b0","dino":0.2}]
    s = summarize(rows)
    assert abs(s["ours"]["dino"]-0.7) < 1e-9
    assert abs(s["b0"]["dino"]-0.2) < 1e-9
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `summarize` (group-by-method mean over numeric keys) and the `evaluate` driver: for each species, build MMKG on `build_refs`, run each method to an image, score every image against `heldout_refs` with all metrics, collect rows; write `results.json` + a markdown table (methods × metrics). **Step 4: Run** unit → PASS.
- [ ] **Step 5: First real run (gpu).** Once `data/` holds Treevill (see Open Items), run:
  `python -m graft.run_eval --root data/<extracted> --n-species 12 --methods ours,b0,b1,b2 --seed 0`
  Confirm `outputs/eval/results.json` and the table are produced; sanity-check that `ours` ≥ `b0` on DINO fidelity for a majority of species. Record the table in the commit body.
- [ ] **Step 6: Commit** `feat(graft): evaluation driver and Phase-1 results`.

---

## Self-Review

**Spec coverage:**
- Spec §3 M1 (visually-grounded MMKG) → Tasks 2,4,5. M2 Lever A → Tasks 6,7. M3 verify-refine → Tasks 8,9. ✓
- Spec §4 metrics (DINOv3/SigLIP2/CLIP-I fidelity, attribute accuracy, CLIP-T) → Task 10; baselines B0/B1/B2 → Task 10; driver + ablations → Task 11. ✓ (Ablations = method flags: B2≈visual-grounding-off; "no IP-Adapter"/"no refine" toggles run via config in the same driver.)
- Spec §5 feasibility (sequential loads, cached ids) → Global Constraints + Task 4 `Models.unload`. ✓
- Spec §6 non-goals → Global Constraints; no task introduces attention surgery/finetuning/compositing. ✓
- Spec §7 deferred (Lever B Kontext, region edits, full benchmark) → intentionally absent. ✓
- Spec §8 Treevill acquisition → Open Items below; anchor stored in M1 (Task 5) though its consumer is Phase 2. ✓

**Placeholder scan:** No "TBD/TODO"; every code step carries real code. ✓
**Type consistency:** `ConceptKG.attribute_texts()` (Task 2) consumed by Tasks 6,8,10; `VerifyReport.ok`/`.failing_parts`/`.part_scores` (Task 8) consumed by Task 9; `Embedder.embed_image` normalized rows consumed by `mean_cosine` (Task 10). Names align across tasks. ✓

## Open Items (carried from spec §8)

- **Treevill:** downloading in background to `kg_test/data/treevill_archive.zip` (2.42 GB). After it lands: `unzip` into `kg_test/data/treevill/`, inspect the top-level folder (`rawdata2/<species>/*.jpg` observed in the stream), and point `--root` at the species-parent directory. The `load_treevill` loader (Task 3) already treats each immediate subdir as a species, so it adapts by choosing the right root.
- **GroundingDINO on botanical parts:** if a part phrase yields no box, M1 (Task 5, step 4) already falls back to whole-ref embedding for that node — no code change needed, but note per-part detection hit-rate in the Task 5 gpu-test output to decide if part phrases need wording tweaks.
