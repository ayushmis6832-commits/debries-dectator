"""
evaluate_anomaly_models.py
-----------------------------
Head-to-head comparison of the two anomaly-detection approaches in this
project:
  - anomaly_detector.py  -- IsolationForest over HOG/LBP features (existing)
  - deep_anomaly.py       -- NumPy autoencoder, reconstruction-error based (new)

Both are trained on the SAME background/normal patches and evaluated on the
SAME held-out test set of real planted objects (should score as anomalous)
vs. held-out normal background patches (should score as normal). This
prints ROC-AUC for each -- the standard metric for "how well does this
score separate the two classes" -- so any claim about which one is more
accurate is a number you can see, not an assertion.

Run from the project root:
    python demo/evaluate_anomaly_models.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.sonar_simulator import generate_scene
from src import preprocessing, ml_classifier, anomaly_detector, deep_anomaly


def collect_patches(n_scenes: int, seed: int):
    """Returns (object_patches, background_patches) from n_scenes fresh scenes."""
    rng = np.random.default_rng(seed)
    object_patches, background_patches = [], []
    for _ in range(n_scenes):
        scene = generate_scene(seed=int(rng.integers(0, 1_000_000)))
        clean = preprocessing.preprocess(scene.image, scene.nadir_cols)
        for obj in scene.objects:
            x0, y0, x1, y1 = obj.bbox()
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(clean.shape[1], x1), min(clean.shape[0], y1)
            if x1 > x0 and y1 > y0:
                object_patches.append(clean[y0:y1, x0:x1])
        for _ in range(4):
            background_patches.append(ml_classifier._sample_negative_patch(clean, scene.nadir_cols, rng, objects=scene.objects))
    return object_patches, background_patches


def main():
    print("=" * 70)
    print("Building TRAINING set (background/normal patches only)")
    print("=" * 70)
    _, train_background = collect_patches(n_scenes=60, seed=1)
    print(f"{len(train_background)} background training patches")

    print("\n" + "=" * 70)
    print("Building HELD-OUT TEST set (fresh scenes, never used in training)")
    print("=" * 70)
    test_objects, test_background = collect_patches(n_scenes=40, seed=999)
    print(f"{len(test_objects)} real object patches (should score as anomalous)")
    print(f"{len(test_background)} background patches (should score as normal)")

    y_true = np.array([1] * len(test_objects) + [0] * len(test_background))  # 1 = anomaly

    print("\n" + "=" * 70)
    print("Training IsolationForest (existing baseline)")
    print("=" * 70)
    t0 = time.time()
    iso_model = anomaly_detector.fit_anomaly_model(train_background)
    iso_train_time = time.time() - t0
    iso_scores = np.array([
        anomaly_detector.anomaly_score(iso_model, p) for p in test_objects + test_background
    ])
    iso_auc = roc_auc_score(y_true, iso_scores)

    print("\n" + "=" * 70)
    print("Training NumPy Autoencoder (new deep-learning approach)")
    print("=" * 70)
    t0 = time.time()
    ae_model = deep_anomaly.train_autoencoder(train_background, epochs=60)
    ae_train_time = time.time() - t0
    ae_scores = np.array([
        deep_anomaly.anomaly_score(ae_model, p) for p in test_objects + test_background
    ])
    ae_auc = roc_auc_score(y_true, ae_scores)

    print("\n" + "=" * 70)
    print("RESULTS (ROC-AUC: 0.5 = no better than random, 1.0 = perfect separation)")
    print("=" * 70)
    print(f"{'Model':<25} {'ROC-AUC':<10} {'Train time':<12}")
    print(f"{'IsolationForest':<25} {iso_auc:<10.4f} {iso_train_time:.2f}s")
    print(f"{'NumPy Autoencoder':<25} {ae_auc:<10.4f} {ae_train_time:.2f}s")

    print()
    if ae_auc > iso_auc + 0.02:
        print(f"-> Autoencoder is more accurate on this test set (+{ae_auc - iso_auc:.3f} AUC).")
    elif iso_auc > ae_auc + 0.02:
        print(f"-> IsolationForest is more accurate on this test set (+{iso_auc - ae_auc:.3f} AUC).")
    else:
        print("-> The two are roughly comparable on this test set (within 0.02 AUC).")
    print("(This is measured on THIS run's held-out data -- rerun with different")
    print(" seeds to check how stable the comparison is before trusting one number.)")


if __name__ == "__main__":
    main()
