import numpy as np
import pandas as pd
import json

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import ProjectConfig
from src.utils import GalaxyVisualizer          

class EvaluateOnUnseenImgs:
    """The ONLY place the test set is used."""

    def __init__(self):
        self.final_model  = None
        self.predictions  = None
        self.metrics      = None

    # ------------------------------------------------------------------
    def RefitModelUsingTrainingAndValidatedDataThenEvaluateOnceOnTheTestSet(
        self, trainer, X_train, y_train, X_val, y_val, X_test, y_test
    ):
        if trainer.chosen_params is None or trainer.chosen_name is None:
            raise RuntimeError(
                "Trainer has no chosen configuration — call "
                "StateVariedParametersAndJustifyFinalChoice() first."
            )

        base_name = trainer.chosen_params["model"]
        kwargs    = {k: v for k, v in trainer.chosen_params.items()
                     if k != "model"}

        X_train = np.asarray(X_train, dtype=np.float32, copy=False)
        X_val   = np.asarray(X_val,   dtype=np.float32, copy=False)
        X_test  = np.asarray(X_test,  dtype=np.float32, copy=False)
        y_train = np.asarray(y_train, dtype=np.int64,   copy=False)
        y_val   = np.asarray(y_val,   dtype=np.int64,   copy=False)
        y_test  = np.asarray(y_test,  dtype=np.int64,   copy=False)

        print("\nFINAL EVALUATION — REFIT ON TRAIN + VALIDATION")
        print("-" * 60)
        print(f"  configuration : {trainer.chosen_name}")
        print(f"  model         : {base_name}")
        print(f"  params        : {kwargs}")
        print(f"  train samples : {len(X_train)}")
        print(f"  val samples   : {len(X_val)}")
        print(f"  test samples  : {len(X_test)}  (evaluated ONCE)")

        # ------------------------------------------------------------------
        # 1. refit on train + val
        # ------------------------------------------------------------------
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)

        self.final_model = trainer.Fit(X_fit, y_fit, base_name, kwargs)

        # ------------------------------------------------------------------
        # 2. predict on the test set exactly once
        # ------------------------------------------------------------------
        self.predictions = self.final_model.predict(X_test)

        # ------------------------------------------------------------------
        # 3. compute + report metrics
        # ------------------------------------------------------------------
        self.metrics = self.ReportAccuracyPerClassPrecisionRecallF1ScoreAndLabeledConfusionMatrix(
            y_test, self.predictions
        )

        print("-" * 60)
        print("Test set consumed. It will not be used again in this run.")

        return self.metrics

    # ------------------------------------------------------------------
    def ReportAccuracyPerClassPrecisionRecallF1ScoreAndLabeledConfusionMatrix(
        self, y_true, y_pred
    ):
        y_true = np.asarray(y_true, dtype=np.int64, copy=False)
        y_pred = np.asarray(y_pred, dtype=np.int64, copy=False)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}."
            )

        labels = list(range(ProjectConfig.NUM_CLASSES))

        # ------------------------------------------------------------------
        # 1. scalar + per-class metrics
        # ------------------------------------------------------------------
        acc = float(accuracy_score(y_true, y_pred))

        precision = precision_score(
            y_true, y_pred,
            labels=labels, average=None, zero_division=0,
        )
        recall = recall_score(
            y_true, y_pred,
            labels=labels, average=None, zero_division=0,
        )
        f1 = f1_score(
            y_true, y_pred,
            labels=labels, average=None, zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # ------------------------------------------------------------------
        # 2. print report
        # ------------------------------------------------------------------
        print("\nTEST-SET METRICS")
        print("-" * 78)
        print(f"Overall accuracy: {acc:.4f}")
        print("-" * 78)
        print(f"{'id':<4}{'class name':<30}{'precision':>11}"
              f"{'recall':>11}{'f1':>11}{'support':>10}")
        print("-" * 78)

        support = cm.sum(axis=1)   # true instances per class
        for cid in labels:
            print(
                f"{cid:<4}"
                f"{ProjectConfig.CLASS_NAMES[cid]:<30}"
                f"{precision[cid]:>11.4f}"
                f"{recall[cid]:>11.4f}"
                f"{f1[cid]:>11.4f}"
                f"{int(support[cid]):>10}"
            )

        # macro / weighted averages
        macro_p  = float(np.mean(precision))
        macro_r  = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(f1_score(y_true, y_pred, labels=labels,
                                     average="weighted", zero_division=0))
        print("-" * 78)
        print(f"{'':<4}{'macro average':<30}"
              f"{macro_p:>11.4f}{macro_r:>11.4f}{macro_f1:>11.4f}"
              f"{int(support.sum()):>10}")
        print(f"{'':<4}{'weighted f1':<30}{'':>11}{'':>11}"
              f"{weighted_f1:>11.4f}")
        print("-" * 78)

        # ------------------------------------------------------------------
        # 3. save metrics to CSV + JSON
        # ------------------------------------------------------------------
        per_class_df = pd.DataFrame({
            "class_id":   labels,
            "class_name": ProjectConfig.CLASS_NAMES,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "support":    support.astype(int),
        })
        per_class_df.to_csv(
            ProjectConfig.METRICS_PATH / "test_per_class_metrics.csv",
            index=False,
        )

        summary = {
            "accuracy":    acc,
            "macro_precision":  macro_p,
            "macro_recall":     macro_r,
            "macro_f1":         macro_f1,
            "weighted_f1":      weighted_f1,
        }
        with open(ProjectConfig.METRICS_PATH / "test_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        # ------------------------------------------------------------------
        # 4. plot labelled confusion matrix
        # ------------------------------------------------------------------
        GalaxyVisualizer.PlotLabeledConfusionMatrix(
            cm, labels=ProjectConfig.CLASS_NAMES
        )

        # ------------------------------------------------------------------
        # 5. return everything the caller may want
        # ------------------------------------------------------------------
        return {
            "accuracy":          acc,
            "precision":         precision,
            "recall":            recall,
            "f1":                f1,
            "macro_precision":   macro_p,
            "macro_recall":      macro_r,
            "macro_f1":          macro_f1,
            "weighted_f1":       weighted_f1,
            "support":           support.astype(int),
            "confusion_matrix":  cm,
        }

    # ------------------------------------------------------------------
    def CompareResultsAgainstBaseline(self, baseline_accuracy, model_accuracy):
        baseline_accuracy = float(baseline_accuracy)
        model_accuracy    = float(model_accuracy)

        if not (0.0 <= baseline_accuracy <= 1.0):
            raise ValueError(
                f"baseline_accuracy out of [0,1]: {baseline_accuracy}"
            )
        if not (0.0 <= model_accuracy <= 1.0):
            raise ValueError(
                f"model_accuracy out of [0,1]: {model_accuracy}"
            )

        abs_gain = model_accuracy - baseline_accuracy
        rel_gain = (
            (abs_gain / baseline_accuracy) * 100.0
            if baseline_accuracy > 0
            else float("inf")
        )

        # verdict: is the model meaningfully better than "predict the mode"?
        if abs_gain <= 0:
            verdict = "MODEL DOES NOT BEAT BASELINE — investigate."
        elif abs_gain < 0.05:
            verdict = "Marginal improvement over baseline (<5 pts)."
        elif abs_gain < 0.20:
            verdict = "Clear improvement over baseline."
        else:
            verdict = "Strong improvement over baseline."

        print("\nMODEL vs MAJORITY-CLASS BASELINE")
        print("-" * 62)
        print(f"{'':<28}{'accuracy':>14}{'vs model':>18}")
        print("-" * 62)
        print(f"{'Majority-class baseline':<28}"
              f"{baseline_accuracy:>14.4f}{'—':>18}")
        print(f"{'Model (final, on test)':<28}"
              f"{model_accuracy:>14.4f}{abs_gain:>+18.4f}")
        print("-" * 62)
        print(f"{'Absolute improvement':<28}{abs_gain:>+14.4f}")
        if baseline_accuracy > 0:
            print(f"{'Relative improvement':<28}{rel_gain:>+13.2f}%")
        else:
            print(f"{'Relative improvement':<28}{'n/a (baseline=0)':>14}")
        print("-" * 62)
        print(f"Verdict: {verdict}")
        print("-" * 62)

        # ------------------------------------------------------------------
        # Persist alongside the other metrics for the report
        # ------------------------------------------------------------------
        row = pd.DataFrame([{
            "baseline_accuracy": baseline_accuracy,
            "model_accuracy":    model_accuracy,
            "absolute_gain":     abs_gain,
            "relative_gain_pct": rel_gain if baseline_accuracy > 0 else None,
            "verdict":           verdict,
        }])
        row.to_csv(
            ProjectConfig.METRICS_PATH / "baseline_comparison.csv",
            index=False,
        )

        return {
            "baseline_accuracy": baseline_accuracy,
            "model_accuracy":    model_accuracy,
            "absolute_gain":     abs_gain,
            "relative_gain_pct": rel_gain if baseline_accuracy > 0 else None,
            "verdict":           verdict,
        }