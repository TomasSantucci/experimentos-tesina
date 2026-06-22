import json
import numpy as np

MAP_PATH = "map_mgo12/opt_map.json"
OUT_PATH = "map_mgo12/opt_map_filtered.json"
# Points whose distance from the centroid exceeds this many MADs are outliers
MAD_THRESHOLD = 10.0


def load_data(path: str):
    with open(path) as f:
        return json.load(f)


def filter_outliers(landmarks: list, threshold: float = MAD_THRESHOLD) -> list:
    pts = np.array([lm["p_w"] for lm in landmarks])
    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    mad = np.median(np.abs(dists - np.median(dists)))
    mask = dists <= np.median(dists) + threshold * mad
    return [lm for lm, keep in zip(landmarks, mask) if keep]


if __name__ == "__main__":
    data = load_data(MAP_PATH)
    landmarks = data["landmarks"]
    print(f"Loaded {len(landmarks)} landmarks")

    filtered = filter_outliers(landmarks)
    print(f"After filtering: {len(filtered)} landmarks ({len(landmarks) - len(filtered)} removed)")

    data["landmarks"] = filtered
    with open(OUT_PATH, "w") as f:
        json.dump(data, f)
    print(f"Saved to {OUT_PATH}")
