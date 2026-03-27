from pathlib import Path
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


_QUALITY_THRESHOLDS = [
    (0.85, "GOOD"),
    (0.70, "OK"),
    (0.50, "WEAK"),
    (0.00, "POOR"),
]


def _quality_tag(f1: float) -> str:
    """Return a quality tag for a given F1 score."""
    for threshold, tag in _QUALITY_THRESHOLDS:
        if f1 >= threshold:
            return tag
    return "POOR"


class EvaluationStrategy:

    def __init__(self, y_true, y_pred, class_labels, model_name="Model"):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.class_labels = list(class_labels)
        self.model_name = model_name

        # Metrics containers
        self.accuracy = None
        self.precision_weighted = None
        self.recall_weighted = None
        self.f1_weighted = None
        self.precision_macro = None
        self.recall_macro = None
        self.f1_macro = None
        self.cm = None
        self.report = None  # keys = class names (target_names)

        self._calculate_metrics()

    def _calculate_metrics(self):
        print(f"\nCalculating metrics for {self.model_name}...")

        self.accuracy = accuracy_score(self.y_true, self.y_pred)

        # Weighted averages
        self.precision_weighted = precision_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.recall_weighted = recall_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.f1_weighted = f1_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0)

        # Macro averages (treats every class equally – critical for imbalanced data)
        self.precision_macro = precision_score(
            self.y_true, self.y_pred, average='macro', zero_division=0)
        self.recall_macro = recall_score(
            self.y_true, self.y_pred, average='macro', zero_division=0)
        self.f1_macro = f1_score(
            self.y_true, self.y_pred, average='macro', zero_division=0)

        # Confusion matrix
        self.cm = confusion_matrix(self.y_true, self.y_pred)

        # Per-class report – keys are class *names* (because target_names is set)
        self.report = classification_report(
            self.y_true, self.y_pred,
            labels=range(len(self.class_labels)),
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0,
        )
        print("Metrics calculated successfully!")

    def calculate_metrics(self) -> Dict[str, Any]:
        return {
            'accuracy': float(self.accuracy),
            'precision_weighted': float(self.precision_weighted),
            'recall_weighted': float(self.recall_weighted),
            'f1_weighted': float(self.f1_weighted),
            'precision_macro': float(self.precision_macro),
            'recall_macro': float(self.recall_macro),
            'f1_macro': float(self.f1_macro),
            'confusion_matrix': self.cm.tolist(),
            'classification_report': {
                k: {mk: float(mv) if isinstance(mv, np.floating) else mv
                    for mk, mv in v.items()} if isinstance(v, dict) else v
                for k, v in self.report.items()
            },
        }

    def print_summary(self) -> None:
        print(f"{self.model_name} - Overall Metrics Summary")
        print(f"  Accuracy:           {self.accuracy:.4f}")
        print(f"  Precision (weighted) {self.precision_weighted:.4f}")
        print(f"  Recall    (weighted) {self.recall_weighted:.4f}")
        print(f"  F1 Score  (weighted) {self.f1_weighted:.4f}")
        print(f"  Precision (macro)    {self.precision_macro:.4f}")
        print(f"  Recall    (macro)    {self.recall_macro:.4f}")
        print(f"  F1 Score  (macro)    {self.f1_macro:.4f}")

    def print_per_class_metrics(self) -> None:
        print("\n" + "=" * 78)
        print(f"{self.model_name} - Per-Class Metrics")
        print("=" * 78)
        header = (f"{'Class':20s} | {'Prec':>7s}  {'Recall':>7s}  "
                  f"{'F1':>7s}  {'Support':>8s}  {'Quality':>7s}")
        print(header)
        print("-" * 78)

        for class_name in self.class_labels:
            if class_name in self.report:
                m = self.report[class_name]
                tag = _quality_tag(m['f1-score'])
                print(f"{class_name:20s} | {m['precision']:7.4f}  "
                      f"{m['recall']:7.4f}  {m['f1-score']:7.4f}  "
                      f"{int(m['support']):>8d}  {tag:>7s}")


    def print_confusion_matrix(self) -> None:
        print(f"{self.model_name} - Confusion Matrix")
        cm_df = pd.DataFrame(
            self.cm,
            index=[f"True: {c}" for c in self.class_labels],
            columns=[f"Pred: {c}" for c in self.class_labels],
        )
        print(cm_df)

    def plot_confusion_matrix(self, figsize=(12, 10), output_path=None):
        plt.figure(figsize=figsize)
        cm_norm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=self.class_labels,
                    yticklabels=self.class_labels,
                    cbar_kws={'label': 'Prediction Rate'})
        plt.title(f"{self.model_name} - Confusion Matrix (Normalized)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        plt.close()

    def plot_per_class_f1(self, output_path=None):
        """Bar chart of per-class F1 with quality-colour coding."""
        names, scores = [], []
        for cn in self.class_labels:
            if cn in self.report:
                names.append(cn)
                scores.append(self.report[cn]['f1-score'])
        colours = []
        for s in scores:
            tag = _quality_tag(s)
            colours.append({'GOOD': '#2ecc71', 'OK': '#f1c40f',
                            'WEAK': '#e67e22', 'POOR': '#e74c3c'}[tag])
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, scores, color=colours, edgecolor='black')
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"{self.model_name} - Per-Class F1 Score")
        for bar, s in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{s:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Per-class F1 chart saved to {output_path}")
        plt.close()

    def save_metrics(self, output_path) -> None:
        metrics_dict = {
            'model_name': self.model_name,
            'overall_metrics': {
                'accuracy': float(self.accuracy),
                'precision_weighted': float(self.precision_weighted),
                'recall_weighted': float(self.recall_weighted),
                'f1_score_weighted': float(self.f1_weighted),
                'precision_macro': float(self.precision_macro),
                'recall_macro': float(self.recall_macro),
                'f1_score_macro': float(self.f1_macro),
            },
            'class_labels': self.class_labels,
            'per_class_metrics': {},
        }
        for class_name in self.class_labels:
            if class_name in self.report:
                m = self.report[class_name]
                metrics_dict['per_class_metrics'][class_name] = {
                    'precision': float(m['precision']),
                    'recall': float(m['recall']),
                    'f1_score': float(m['f1-score']),
                    'support': int(m['support']),
                    'quality': _quality_tag(m['f1-score']),
                }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {output_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'accuracy': float(self.accuracy),
            'f1_weighted': float(self.f1_weighted),
            'f1_macro': float(self.f1_macro),
            'precision_macro': float(self.precision_macro),
            'recall_macro': float(self.recall_macro),
            'num_classes': len(self.class_labels),
        }


# ══════════════════════════════════════════════════════════════════════════
#  Model Comparison
# ══════════════════════════════════════════════════════════════════════════

def compare_models(evaluations: List[EvaluationStrategy],
                   output_dir: Path = None) -> pd.DataFrame:
    """
    Print a side-by-side comparison table for multiple models and
    optionally save a comparison chart.
    """
    rows = []
    for ev in evaluations:
        rows.append({
            'Model': ev.model_name,
            'Accuracy': ev.accuracy,
            'F1 (macro)': ev.f1_macro,
            'F1 (weighted)': ev.f1_weighted,
            'Prec (macro)': ev.precision_macro,
            'Rec (macro)': ev.recall_macro,
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False, float_format='{:.4f}'.format))
    print("=" * 80)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save CSV
        df.to_csv(output_dir / "model_comparison.csv", index=False)
        # Save grouped bar chart
        metrics_cols = [c for c in df.columns if c != 'Model']
        x = np.arange(len(metrics_cols))
        width = 0.8 / len(evaluations)
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row[c] for c in metrics_cols]
            ax.bar(x + i * width, vals, width, label=row['Model'])
        ax.set_xticks(x + width * (len(evaluations) - 1) / 2)
        ax.set_xticklabels(metrics_cols, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison chart saved to {output_dir / 'model_comparison.png'}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Convenience functions
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, class_labels,
                   model_name: str = "Model") -> EvaluationStrategy:
    """Evaluate a trained model and return an EvaluationStrategy object."""
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    return EvaluationStrategy(y_true=y_test, y_pred=y_pred,
                              class_labels=class_labels,
                              model_name=model_name)


def generate_evaluation_report(evaluation_strategy: EvaluationStrategy,
                               output_dir: Path = None,
                               model_name: str = "Model") -> None:
    """Generate complete evaluation report with visualizations."""
    print("\n" + "=" * 70)
    print(f"Generating {model_name} Evaluation Report")
    print("=" * 70)

    evaluation_strategy.print_summary()
    evaluation_strategy.print_per_class_metrics()
    evaluation_strategy.print_confusion_matrix()

    if output_dir:
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        slug = model_name.lower().replace(' ', '_')

        evaluation_strategy.plot_confusion_matrix(
            output_path=figures_dir / f"{slug}_confusion_matrix.png")
        evaluation_strategy.plot_per_class_f1(
            output_path=figures_dir / f"{slug}_per_class_f1.png")
        evaluation_strategy.save_metrics(output_dir / f"{slug}_metrics.json")
        print(f"\nReport files saved to: {output_dir}")

    print("\nEvaluation report generation completed!")
