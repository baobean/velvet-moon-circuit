"""Protocols every GRAFT module depends on, never on concrete model classes.

Logic modules (kg_build, generate, verify, refine, metrics, baselines)
receive instances implementing these Protocols as arguments, so their pure
decision logic is unit-testable with in-memory fakes -- only graft.models
touches HuggingFace weights.
"""
from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image


class VLM(Protocol):
    def describe(self, images: Sequence[Image.Image], instruction: str) -> str:
        """Free-form reply to `instruction`, grounded in `images`."""
        ...

    def ask(self, image: Image.Image, question: str) -> str:
        """Free-form reply to a yes/no-style `question` about one image."""
        ...


class Reranker(Protocol):
    def rank(self, query: str, images: Sequence[Image.Image]) -> List[float]:
        """One relevance score per image, higher = more relevant to query."""
        ...


class Embedder(Protocol):
    def embed_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        """L2-normalized image embeddings, shape (N, D)."""
        ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized text embeddings, shape (N, D)."""
        ...


class Detector(Protocol):
    def detect(
        self, image: Image.Image, phrase: str
    ) -> List[Tuple[float, float, float, float]]:
        """Boxes (xyxy, pixel coords) for `phrase` in `image`, best-first."""
        ...


class Segmenter(Protocol):
    def mask(self, image: Image.Image, box: Tuple[float, float, float, float]) -> np.ndarray:
        """Boolean HxW mask for the object inside `box`."""
        ...


class Generator(Protocol):
    def generate(
        self,
        prompt: str,
        ip_image: Image.Image | None,
        seed: int,
        negative: str,
        steps: int,
    ) -> Image.Image:
        """One SDXL(+IP-Adapter) sample."""
        ...
