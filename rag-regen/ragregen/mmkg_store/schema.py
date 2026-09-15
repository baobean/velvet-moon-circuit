"""MMKG species store schema — record shape, id/hub helpers, attribute reshape."""


def global_id(dataset: str, species_key: str) -> str:
    """Generate global ID from dataset and species key.

    Args:
        dataset: Dataset identifier
        species_key: Species key within dataset

    Returns:
        Namespaced ID string: "<dataset>:<species_key>"
    """
    return f"{dataset}:{species_key}"


def hub_id(dataset: str, rank: str, name: str) -> str:
    """Generate hub ID for taxonomy ranks (genus, family).

    Args:
        dataset: Dataset identifier
        rank: Taxonomic rank ("genus", "family")
        name: Taxon name

    Returns:
        Namespaced hub ID string: "<dataset>:<rank>:<name>"
    """
    return f"{dataset}:{rank}:{name}"


def attribute_record(consensus: dict, source: str) -> dict:
    """Extract attribute records from consensus, filtering for is_target slots only.

    Processes bird_target_attributes output, returning only slots where is_target==True.
    Each slot retains value, support, and visible_count (both may be null), with source added.

    Args:
        consensus: Dict of {slot: {value, support, visible_count, is_target, ...}}
        source: Source identifier to include in output records

    Returns:
        Dict of {slot: {value, support, visible_count, source}} for is_target==True slots only.
    """
    result = {}
    for slot, record in consensus.items():
        if record.get("is_target", False):
            result[slot] = {
                "value": record["value"],
                "support": record["support"],
                "visible_count": record["visible_count"],
                "source": source,
            }
    return result


REQUIRED_MEDOID_KEYS = {"image_path", "embedding_ref", "k_images", "selection"}


def build_record(
    *,
    dataset: str,
    species_key: str,
    scientific_name: str,
    common_name: str | None,
    taxonomy: dict,
    medoid: dict,
    candidates: list,
    part_crops: list,
    attributes: dict,
    provenance: dict,
) -> dict:
    """Build complete species record with validation.

    Assembles record with global_id, validates required keys, and constructs namespaced
    relations (instance_of) from taxonomy (genus and family).

    Args:
        dataset: Dataset identifier (must not be empty)
        species_key: Species key (must not be empty)
        scientific_name: Scientific name
        common_name: Common name (may be None)
        taxonomy: Dict with "genus" and "family" keys
        medoid: Dict with image_path (must exist and not be empty) and other keys
        candidates: List of candidate images
        part_crops: List of part crop records
        attributes: Dict of attributes
        provenance: Dict of provenance info

    Returns:
        Record dict with global_id, common_name, relations, and other fields

    Raises:
        ValueError: If medoid.image_path is missing/empty or dataset/species_key empty
    """
    # Validate required keys
    if not dataset or not species_key:
        raise ValueError("dataset and species_key must not be empty")

    if "image_path" not in medoid or not medoid["image_path"]:
        raise ValueError("medoid.image_path must exist and not be empty")

    # Build relations with namespaced instance_of
    relations = []
    if "genus" in taxonomy:
        relations.append({
            "type": "instance_of",
            "target": hub_id(dataset, "genus", taxonomy["genus"]),
        })
    if "family" in taxonomy:
        relations.append({
            "type": "instance_of",
            "target": hub_id(dataset, "family", taxonomy["family"]),
        })

    # Assemble record
    record = {
        "global_id": global_id(dataset, species_key),
        "dataset": dataset,
        "species_key": species_key,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "taxonomy": taxonomy,
        "medoid": medoid,
        "candidates": candidates,
        "part_crops": part_crops,
        "attributes": attributes,
        "provenance": provenance,
        "relations": relations,
    }

    return record
