from graft.worker_build_heldout_parts import heldout_record


def test_heldout_record_shape():
    r = heldout_record("Bamboo", "leaf", "Bamboo/r7.jpg", (1.0, 2.0, 3.0, 4.0),
                       "m.npy", "c.png", [0.1, 0.2])
    assert r == {"concept": "Bamboo", "part": "leaf", "ref_path": "Bamboo/r7.jpg",
                 "box": [1, 2, 3, 4], "mask_path": "m.npy", "crop_path": "c.png",
                 "siglip2": [0.1, 0.2]}
