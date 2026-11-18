import time
import pandas as pd
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from config import ProjectConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class TrainClassifiers:
    """
    Compares several sklearn estimators on (X_train, y_train) using
    (X_val, y_val) accuracy as the selection signal. Test set is NOT
    touched here — that is Task E.
    """

    def __init__(self):
        self.fitted_models  = {}      # name -> fitted estimator
        self.val_results    = {}      # name -> {"val_accuracy": ..., "params": {...}}
        self.chosen_name    = None
        self.chosen_params  = None

    # ------------------------------------------------------------------
    def Fit(self, X_train, y_train, name, params=None):
        """
        Fit a single named configuration. Returns the fitted estimator.

        Linear, kernel, and distance-based models are wrapped in a
        StandardScaler pipeline: raw pixel features in [0, 1] have
        per-column variances that differ by orders of magnitude, which
        biases the SVM / logistic regulariser. Scaling first makes the
        C sweep invert — larger C wins, accuracy jumps ~20 points.

        Tree models (RandomForest) are not scaled — split thresholds are
        invariant to feature scale.
        """
        params = params or {}
        X_train = np.asarray(X_train, dtype=np.float32, copy=False)
        y_train = np.asarray(y_train, dtype=np.int64, copy=False)

        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2-D; got {X_train.shape}.")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X/y length mismatch: {len(X_train)} vs {len(y_train)}."
            )

        S = ProjectConfig.RANDOM_SEED

        # Prototypes: fixed kwargs only; user params applied via
        # set_params() AFTER validation so a wrong key fails loudly.
        prototypes = {
            "linear_svc": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(random_state=S, max_iter=5000)),
            ]),
            "rbf_svc": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", random_state=S)),
            ]),
            "logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=S)),
            ]),
            "knn": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            "random_forest": RandomForestClassifier(
                n_jobs=ProjectConfig.N_JOBS,
                random_state=S,
            ),
        }

        if name not in prototypes:
            raise ValueError(
                f"Unknown estimator '{name}'. "
                f"Choices: {sorted(prototypes)}."
            )

        base = prototypes[name]

        # ------------------------------------------------------------------
        # Validate params against what the estimator actually accepts.
        # Pipelines expose params as clf__C, clf__max_iter, etc. — we
        # validate against the inner estimator and prefix on the way in.
        # ------------------------------------------------------------------
        if isinstance(base, Pipeline):
            accepted   = set(base.named_steps["clf"].get_params().keys())
            unexpected = set(params) - accepted
            if unexpected:
                raise ValueError(
                    f"Estimator '{name}' does not accept parameter(s) "
                    f"{sorted(unexpected)}. "
                    f"Likely cause: a params dict from a different model "
                    f"was reused. Accepted keys (sample): "
                    f"{sorted(accepted)[:10]}…"
                )
            model = base.set_params(**{f"clf__{k}": v for k, v in params.items()})
        else:
            accepted   = set(base.get_params().keys())
            unexpected = set(params) - accepted
            if unexpected:
                raise ValueError(
                    f"Estimator '{name}' does not accept parameter(s) "
                    f"{sorted(unexpected)}. "
                    f"Likely cause: a params dict from a different model "
                    f"was reused. Accepted keys (sample): "
                    f"{sorted(accepted)[:10]}…"
                )
            model = base.set_params(**params)

        model.fit(X_train, y_train)
        self.fitted_models[name] = model
        return model

    # ------------------------------------------------------------------
    def CompareAtLeast3ConfigsUsingValidationPerformance(self, X_train, y_train, X_val, y_val):
        X_train = np.asarray(X_train, dtype=np.float32, copy=False)
        X_val   = np.asarray(X_val,   dtype=np.float32, copy=False)
        y_train = np.asarray(y_train, dtype=np.int64,   copy=False)
        y_val   = np.asarray(y_val,   dtype=np.int64,   copy=False)

        # CPU-friendly grid: no rbf_svc here — at 3072-D features its
        # O(n^2) kernel matrix blows both time and RAM. RBF is still
        # available via Fit() if a user explicitly requests it.
        grid = {
            "linear_svc_C0.01":   ("linear_svc",    {"C": 0.01}),
            "linear_svc_C0.1":    ("linear_svc",    {"C": 0.1}),
            "linear_svc_C1.0":    ("linear_svc",    {"C": 1.0}),
            "logistic_C1.0":      ("logistic",      {"C": 1.0}),
            "random_forest_200":  ("random_forest", {"n_estimators": 200}),
        }

        self.val_results = {}

        print("\nVALIDATION-TIME CONFIGURATION COMPARISON")
        print("-" * 70)
        print(f"{'config':<22}{'val_acc':>10}{'fit_s':>10}{'pred_s':>10}")
        print("-" * 70)

        best_name  = None
        best_acc   = -1.0

        for label, (name, params) in grid.items():
            t0 = time.perf_counter()
            model = self.Fit(X_train, y_train, name, params)
            t_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict(X_val)
            t_pred = time.perf_counter() - t0

            acc = float(accuracy_score(y_val, pred))

            self.val_results[label] = {
                "val_accuracy": acc,
                "params":       {"model": name, **params},
                "fit_seconds":  round(t_fit, 2),
                "pred_seconds": round(t_pred, 2),
            }

            print(f"{label:<22}{acc:>10.4f}{t_fit:>10.2f}{t_pred:>10.2f}")

            if acc > best_acc:
                best_acc  = acc
                best_name = label

        print("-" * 70)
        print(f"Best on validation: {best_name}  (acc={best_acc:.4f})")

        # Persist comparison for the report.
        rows = []
        for label, r in self.val_results.items():
            rows.append({
                "config":       label,
                "model":        r["params"]["model"],
                "val_accuracy": r["val_accuracy"],
                "fit_seconds":  r["fit_seconds"],
                "pred_seconds": r["pred_seconds"],
                **{k: v for k, v in r["params"].items() if k != "model"},
            })
        pd.DataFrame(rows).to_csv(
            ProjectConfig.REPORTS_PATH / "classifier_comparison.csv",
            index=False,
        )

        return self.val_results

    # ------------------------------------------------------------------
    def StateVariedParametersAndJustifyFinalChoice(self):
        if not self.val_results:
            raise RuntimeError(
                "No validation results — call "
                "CompareAtLeast3ConfigsUsingValidationPerformance() first."
            )

        # ------------------------------------------------------------------
        # 1. show the grid that was actually varied
        # ------------------------------------------------------------------
        print("\nPARAMETERS VARIED DURING COMPARISON")
        print("-" * 65)
        print(f"{'config':<22}{'model':<16}{'params':<25}")
        print("-" * 65)
        for label, r in self.val_results.items():
            model  = r["params"].get("model", "?")
            extras = {k: v for k, v in r["params"].items() if k != "model"}
            print(f"{label:<22}{model:<16}{str(extras):<25}")
        print("-" * 65)

        # ------------------------------------------------------------------
        # 2. pick the winner by validation accuracy
        # ------------------------------------------------------------------
        best_label = max(
            self.val_results,
            key=lambda k: self.val_results[k]["val_accuracy"],
        )
        best_acc  = self.val_results[best_label]["val_accuracy"]
        runner_up = sorted(
            self.val_results,
            key=lambda k: self.val_results[k]["val_accuracy"],
            reverse=True,
        )[1]
        runner_acc = self.val_results[runner_up]["val_accuracy"]

        self.chosen_name   = best_label
        self.chosen_params = self.val_results[best_label]["params"]

        # ------------------------------------------------------------------
        # 3. print the choice + justification
        # ------------------------------------------------------------------
        print("\nFINAL CHOICE")
        print("-" * 65)
        print(f"  chosen configuration : {best_label}")
        print(f"  underlying model     : {self.chosen_params['model']}")
        print(f"  parameters           : "
              f"{ {k: v for k, v in self.chosen_params.items() if k != 'model'} }")
        print(f"  validation accuracy  : {best_acc:.4f}")
        print(f"  runner-up            : {runner_up}  ({runner_acc:.4f})")
        print(f"  margin               : {best_acc - runner_acc:+.4f}")
        print("-" * 65)

        print(
            f"\nJustification:\n"
            f"  '{best_label}' achieved the highest validation accuracy "
            f"({best_acc:.4f}) across {len(self.val_results)} configurations, "
            f"beating the runner-up '{runner_up}' by "
            f"{(best_acc - runner_acc) * 100:.2f} percentage points. "
            f"Model selection used ONLY the validation split — the test set "
            f"remained sealed — so this choice is leakage-free. "
            f"The winner is now refit on train+validation in Task E and "
            f"evaluated exactly once on the untouched test set."
        )

        return self.chosen_name, self.chosen_params