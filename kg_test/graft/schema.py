"""Multimodal knowledge graph (MMKG) data model for a single rare concept.

Every value on a node is either read from a real reference image ("vision")
or filled from the VLM's parametric memory ("llm") -- attribute_texts() lists
vision-sourced values first so downstream prompts and metrics prefer
verifiable evidence over recall.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class AttributeNode:
    name: str
    value: str
    source: str  # "vision" | "llm"


@dataclass
class PartNode:
    name: str
    attributes: List[AttributeNode]
    exemplar_crop: str | None
    embeddings: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ConceptKG:
    concept: str
    global_attrs: List[AttributeNode]
    parts: List[PartNode]
    concept_embeddings: Dict[str, List[float]]
    anchor: str
    delta: List[str]
    ref_paths: List[str]
    ref_embeddings: List[List[float]] = field(default_factory=list)  # SigLIP2, aligned to ref_paths

    def attribute_texts(self) -> List[str]:
        """Flatten global + part attributes into phrases, vision-sourced first."""

        def is_vision(a: AttributeNode) -> bool:
            return a.source == "vision"

        global_sorted = sorted(self.global_attrs, key=lambda a: not is_vision(a))
        texts = [f"{a.name}: {a.value}" for a in global_sorted]

        part_attrs = []
        for part in self.parts:
            for a in part.attributes:
                part_attrs.append((a, f"{part.name} {a.name}: {a.value}"))
        part_attrs.sort(key=lambda pair: not is_vision(pair[0]))
        texts.extend(text for _, text in part_attrs)
        return texts

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ConceptKG":
        with open(path) as f:
            raw = json.load(f)
        global_attrs = [AttributeNode(**a) for a in raw["global_attrs"]]
        parts = [
            PartNode(
                name=p["name"],
                attributes=[AttributeNode(**a) for a in p["attributes"]],
                exemplar_crop=p["exemplar_crop"],
                embeddings=p.get("embeddings", {}),
            )
            for p in raw["parts"]
        ]
        return cls(
            concept=raw["concept"],
            global_attrs=global_attrs,
            parts=parts,
            concept_embeddings=raw.get("concept_embeddings", {}),
            anchor=raw["anchor"],
            delta=raw["delta"],
            ref_paths=raw["ref_paths"],
            ref_embeddings=raw.get("ref_embeddings", []),
        )
