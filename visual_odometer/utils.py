DEFAULT_CONFIG = {
    "Displacement Estimation": {
        "method": "projection-svd",
        "reprocess_displacement": True,
        "skip_frames": False,
        "params": {
            "skip_frames_threshold": 5,
            "reprocess_displacement_count": 1
        },
    },
    "Frequency Window": {
        "method": "Stone_et_al_2001",
        "params": {
            "factor": 0.6,
        }
    },
    "Spatial Window": {
        "method": "raised_cosine",
        "params": {
            "a0": 0.358,
            "a1": 0.47,
            "a2": 0.135,
            "a3": 0.037,
        }
    },
    "Downsampling": {
        "method": "",
        "params": {
            "factor": 1,
        }
    },
}

def merge_dicts(base: dict, override: dict):
    """Recursively merges two dictionaries."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base

