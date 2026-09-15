from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field

@dataclass
class PartInstanceRec:
    id: str
    concept: str
    part: str
    ref_path: str
    crop_path: str
    siglip2: list[float]

@dataclass
class PartTypeRec:
    id: int
    part: str
    member_ids: list[str]
    centroid: list[float]
    coherence: float
    label: str = ""

@dataclass
class PartTypeGraph:
    part_instances: list[PartInstanceRec] = field(default_factory=list)
    part_types: list[PartTypeRec] = field(default_factory=list)

    def instances_by_id(self) -> dict[str, PartInstanceRec]:
        return {i.id: i for i in self.part_instances}

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"part_instances": [asdict(i) for i in self.part_instances],
                       "part_types": [asdict(h) for h in self.part_types]}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PartTypeGraph":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            part_instances=[PartInstanceRec(**i) for i in raw["part_instances"]],
            part_types=[PartTypeRec(**h) for h in raw["part_types"]],
        )
