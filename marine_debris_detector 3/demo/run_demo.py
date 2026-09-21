"""
run_demo.py
------------
One-command demo of the full prototype:

  1. Trains the RandomForest debris classifier on synthetic labeled sonar scenes.
  2. Trains the NumPy-autoencoder anomaly model on background seabed patches
     (a real neural network, trained via backpropagation -- see deep_anomaly.py
     for why this is implemented without a PyTorch/TensorFlow dependency).
  3. Generates a fresh, never-seen synthetic sonar scene ("survey line").
  4. Runs the full detection pipeline on it.
  5. Saves an annotated detection image, a ground-truth comparison image,
     and a JSON detection report to outputs/.

Run from the project root:
    python demo/run_demo.py

For a measured comparison of this autoencoder against the IsolationForest
baseline it replaced as the default, run:
    python demo/evaluate_anomaly_models.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from src import preprocessing, ml_classifier, deep_anomaly
from src.sonar_simulator import generate_scene, to_uint8
from src.pipeline import MarineDebrisPipeline
from src.map_export import export_map
from src.reporting import report_to_csv
from src import segmentation

OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CLF_PATH = str(MODEL_DIR / "debris_classifier.joblib")
ANOM_PATH = str(MODEL_DIR / "anomaly_model_autoencoder.joblib")  # see dashboard/app.py for why
                                                                   # this filename changed


def main():
    print("=" * 60)
    print("STEP 1/4  Training supervised debris classifier on synthetic data")
    print("=" * 60)
    ml_classifier.train(n_scenes=280, seed=42, model_path=CLF_PATH)

    print("\n" + "=" * 60)
    print("STEP 2/4  Training NumPy autoencoder anomaly model (deep learning,")
    print("          reconstruction-error based -- see deep_anomaly.py)")
    print("=" * 60)
    rng = np.random.default_rng(7)
    bg_patches = []
    for _ in range(60):
        scene = generate_scene(seed=int(rng.integers(0, 1_000_000)))
        clean = preprocessing.preprocess(scene.image, scene.nadir_cols)
        for _ in range(3):
            bg_patches.append(ml_classifier._sample_negative_patch(clean, scene.nadir_cols, rng, objects=scene.objects))
    anom_model = deep_anomaly.train_autoencoder(bg_patches)
    deep_anomaly.save(anom_model, ANOM_PATH)
    print(f"anomaly model saved -> {ANOM_PATH}")

    print("\n" + "=" * 60)
    print("STEP 3/4  Generating a fresh test survey line (never seen in training)")
    print("=" * 60)
    test_scene = generate_scene(seed=99999, n_objects=6)
    print(f"planted {len(test_scene.objects)} ground-truth objects: "
          f"{[o.cls for o in test_scene.objects]}")

    print("\n" + "=" * 60)
    print("STEP 4/4  Running detection pipeline")
    print("=" * 60)
    pipeline = MarineDebrisPipeline(
        classifier_model_path=CLF_PATH,
        anomaly_model_path=ANOM_PATH,
        classifier_conf_threshold=0.35,
        anomaly_threshold=0.55,
    )
    detections, annotated = pipeline.run(test_scene.image, test_scene.nadir_cols, geo=test_scene.geo)
    report = pipeline.report(detections)
    print(f"{report['num_detections']} detections in {report['runtime_seconds']}s")
    for d in report["detections"]:
        print(f"  - {d['predicted_class']:15s} conf={d['classifier_confidence']:.2f} "
              f"anomaly={d['anomaly_score']:.2f} bbox={d['bbox']} "
              f"lat={d['lat']} lon={d['lon']}")

    # ---- segmentation quality: IoU of predicted masks vs ground-truth instance mask ----
    print("\n" + "=" * 60)
    print("Segmentation quality (predicted mask vs. ground-truth mask)")
    print("=" * 60)
    ious = []
    for d in detections:
        if not d.mask_polygon:
            continue
        pred_mask = segmentation.polygon_to_mask(d.mask_polygon, test_scene.image.shape)
        x0, y0, x1, y1 = d.bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        gt_id = test_scene.instance_mask[cy, cx] if 0 <= cy < test_scene.image.shape[0] and 0 <= cx < test_scene.image.shape[1] else 0
        if gt_id == 0:
            continue  # this detection's center isn't over a real planted object (likely a false alarm)
        gt_mask = test_scene.instance_mask == gt_id
        iou = segmentation.mask_iou(pred_mask, gt_mask)
        ious.append(iou)
        print(f"  {d.predicted_class:15s} bbox={d.bbox}  IoU vs ground truth = {iou:.2f}")
    if ious:
        print(f"\nmean IoU over {len(ious)} matched detections: {np.mean(ious):.3f}")
    else:
        print("no detections matched a ground-truth object's location")

    # ---- save outputs ----
    raw_u8 = to_uint8(test_scene.image)
    cv2.imwrite(str(OUT_DIR / "01_raw_sonar.png"), raw_u8)

    gt_vis = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
    for obj in test_scene.objects:
        x0, y0, x1, y1 = obj.bbox()
        cv2.rectangle(gt_vis, (x0, y0), (x1, y1), (255, 180, 0), 1)
        cv2.putText(gt_vis, obj.cls, (x0, max(10, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "02_ground_truth.png"), gt_vis)
    cv2.imwrite(str(OUT_DIR / "03_detections.png"), annotated)

    with open(OUT_DIR / "detection_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(OUT_DIR / "detection_report.csv", "w") as f:
        f.write(report_to_csv(report))

    map_path = export_map(test_scene.geo, report["detections"], str(OUT_DIR / "04_map.html"))

    print(f"\nSaved outputs to {OUT_DIR}")
    print("  01_raw_sonar.png        - simulated raw sonar waterfall")
    print("  02_ground_truth.png     - planted objects (for evaluation only)")
    print("  03_detections.png       - pipeline's predicted detections")
    print("  detection_report.json   - structured detection report (incl. lat/lon per detection)")
    print("  detection_report.csv    - flat CSV version of the same report")
    print("  04_map.html             - survey track + geo-tagged detections (open in a browser)")
    if report["num_dropout_rows_repaired"]:
        print(f"  note: {report['num_dropout_rows_repaired']} motion-dropout ping(s) were "
              f"detected and repaired before detection: {report['dropout_rows_repaired']}")


if __name__ == "__main__":
    main()
