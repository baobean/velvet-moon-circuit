"""Reference source adapter — precomputed medoid retrieval (no FAISS, no NN fallback)."""
from PIL import Image


def medoid_reference(store, global_id):
    """Fetch medoid image for a species by global_id.

    Args:
        store: Store with get(global_id) method.
        global_id: Species identifier.

    Returns:
        PIL.Image in RGB mode, or None if species not found.
    """
    rec = store.get(global_id)
    if rec is None:
        return None
    img = Image.open(rec["medoid"]["image_path"])
    return img.convert("RGB")
