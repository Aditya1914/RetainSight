"""
One-command setup: generate data + train all models.
Run this before launching the dashboard.

Usage:
  python setup.py           # random seed each run → different data each time
  python setup.py 42        # fixed seed → reproducible results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_generation.generate import generate_all
from src.ml.models import train_churn_model, train_segmentation, train_ltv_model


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  RetainSight — Setup")
    print("=" * 60)

    print("\n[1/4] Generating synthetic dataset...")
    generate_all(seed=seed)

    print("\n[2/4] Training churn prediction model...")
    churn_results = train_churn_model()
    print(f"  Best model: {churn_results['best_model']}")
    for name, metrics in churn_results["results"].items():
        print(f"  {name}: AUC={metrics['auc_roc']}, F1={metrics['f1']}")

    print("\n[3/4] Training customer segmentation...")
    seg_results = train_segmentation()
    print(f"  Silhouette score: {seg_results['silhouette_score']}")
    print(f"  Segments:\n{seg_results['profiles'].to_string(index=False)}")

    print("\n[4/4] Training LTV model...")
    ltv_results = train_ltv_model()
    print(f"  MAE: ${ltv_results['mae']:.2f}")
    print(f"  R²:  {ltv_results['r2']}")

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Run the dashboard: streamlit run app/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
