import os
_GB = os.environ.get("GALAXY10_MEM_GB")
if _GB:
    try:
        import resource
        _lim = int(float(_GB) * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (_lim, _lim))
        print(f"[memory] RLIMIT_AS = {_GB} GB")
    except (ImportError, ValueError, OSError):
        pass

import traceback
import numpy as np

from config import ProjectConfig
from src.utils import set_global_seed, apply_memory_limit
from src.exploration     import ExploreData
from src.processing.process_image import ProcessImage
from src.experimentation import DesignExperiment
from src.training        import TrainClassifiers
from src.evaluation      import EvaluateOnUnseenImgs
from src.interpretation  import InterpretErrors

from src.utils import configure_cpu_only_environment
from src.utils import set_global_seed, apply_memory_limit, GalaxyVisualizer
configure_cpu_only_environment()

# ----------------------------------------------------------------------
# Terminal helpers
# ----------------------------------------------------------------------
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def terminal_pause():
    input("\nPress ENTER to return to the main menu...")


def terminal_header():
    clear_terminal()
    print("=" * 65)
    print("            GALAXY10 — INTERACTIVE RESEARCH TERMINAL")
    print("=" * 65)
    print("       Pixel-Based Galaxy Classification (DECals)")
    print("=" * 65)

import pickle

SESSION_FILE = ProjectConfig.CHECKPOINT_PATH / "session.pkl"


def save_state(state, pipeline):
    """Persist the interactive session to disk so it can be resumed."""
    snapshot = {
        "state": {
            k: v for k, v in state.items() if k != "images"
        },
        # Trainer state — needed to reach Task E after a reload
        "trainer_chosen_name":   pipeline.trainer.chosen_name,
        "trainer_chosen_params": pipeline.trainer.chosen_params,
        "trainer_val_results":   pipeline.trainer.val_results,
        # Evaluator state — needed to reach Task F after a reload
        "evaluator_predictions": pipeline.evaluator.predictions,
        "evaluator_metrics":     pipeline.evaluator.metrics,
    }
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_FILE, "wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = SESSION_FILE.stat().st_size / 1e6
    print(f"\n[session saved] {SESSION_FILE}  ({size_mb:.1f} MB)")


def load_state(state, pipeline):
    """Restore the interactive session from disk. Returns True on success."""
    if not SESSION_FILE.exists():
        return False
    with open(SESSION_FILE, "rb") as f:
        snapshot = pickle.load(f)

    # state dict
    for k, v in snapshot["state"].items():
        if k in state:
            state[k] = v

    # pipeline objects
    pipeline.trainer.chosen_name   = snapshot["trainer_chosen_name"]
    pipeline.trainer.chosen_params = snapshot["trainer_chosen_params"]
    pipeline.trainer.val_results   = snapshot["trainer_val_results"]
    pipeline.evaluator.predictions = snapshot["evaluator_predictions"]
    pipeline.evaluator.metrics     = snapshot["evaluator_metrics"]

    print(f"\n[session loaded] {SESSION_FILE}")
    return True

def terminal_status(state):
    print("\nEXPERIMENT PROGRESS")
    print("-" * 45)
    stages = [
        "Data Exploration",
        "Image Processing",
        "Experimental Design",
        "Classifier Training",
        "Final Evaluation",
        "Error Interpretation",
    ]
    for s in stages:
        tag = "[DONE]" if state.get(s) else "[    ]"
        print(f"{tag}  {s}")


# ----------------------------------------------------------------------
# Pipeline facade
# ----------------------------------------------------------------------
class Galaxy10Pipeline:
    """Holds one instance of each task class. No side effects on init."""

    def __init__(self):
        ProjectConfig.create_directories()
        self.explorer      = ExploreData()
        self.processor     = ProcessImage()
        self.experiment    = DesignExperiment()
        self.trainer       = TrainClassifiers()
        self.evaluator     = EvaluateOnUnseenImgs()
        self.interpreter   = InterpretErrors()

# ---------------------------------------------------------------------
# Auto Resume
# ---------------------------------------------------------------------
'''pipeline = Galaxy10Pipeline()

# ----- auto-resume prompt -----
if SESSION_FILE.exists():
    ans = input(
    f"Found saved session ({SESSION_FILE.name}). Restore? [y/N]: "
    ).strip().lower()
    if ans == "y":
        try:
            load_state(state, pipeline)
        except Exception as err:
            print(f"[warn] could not restore session: {err}")'''

# ----------------------------------------------------------------------
# Interactive session
# ----------------------------------------------------------------------
def interactive_terminal():
    set_global_seed(ProjectConfig.RANDOM_SEED)
    apply_memory_limit(ProjectConfig.MEMORY_LIMIT_GB)

    pipeline = Galaxy10Pipeline()

    state = {
        "Data Exploration":     False,
        "Image Processing":     False,
        "Experimental Design":  False,
        "Classifier Training":  False,
        "Final Evaluation":     False,
        "Error Interpretation": False,
        # buffers
        "images": None, "labels": None, "indices": None,
        "processed_images": None, "features": None,
        "train_idx": None, "val_idx": None, "test_idx": None,
        "training_results": None, "evaluation_results": None,
    }

    # ---- session restore: pickle first, .npy fallback ----   ← 3rd
    restored = False
    if SESSION_FILE.exists():
        ans = input(
            f"\nFound saved session ({SESSION_FILE.name}, "
            f"{SESSION_FILE.stat().st_size / 1e6:.0f} MB). Restore? [y/N]: "
        ).strip().lower()
        if ans == "y":
            try:
                restored = load_state(state, pipeline)
            except Exception as err:
                print(f"[warn] pickle restore failed: {err}")

    if not restored:
        try:
            restored = try_load_from_disk(state, pipeline)
        except Exception as err:
            print(f"[warn] .npy cache load failed: {err}")

    if restored:
        terminal_pause()

    def run_task(name, fn):
        print(f"\n>>> {name}")
        print("-" * 55)
        try:
            fn()
            state[name] = True
            print(f"\n[SUCCESS] {name} completed.")
            save_state(state, pipeline)          # ← new
        except NotImplementedError:
            print(f"\n[STUB] {name} is a scaffold — fill in the TODO block.")
        except Exception as err:
            print(f"\n[ERROR] {name} failed: {err}")
            traceback.print_exc()
        terminal_pause()

    
    # ---------- Task A ----------
    def task_exploration():
        images, labels = pipeline.explorer.LoadDataset()
        pipeline.explorer.ReportOriginalClassDistribution()
        pipeline.explorer.ApplyPrescribedClassMapping()
        pipeline.explorer.DisplayAtLeast12RepresentativeImagesAndDescribeVisibleDifferences()
        images, labels, indices = pipeline.explorer.SelectABalancedSubset()
        state["images"], state["labels"], state["indices"] = images, labels, indices
        print("Balanced dataset:", images.shape)

    # ---------- Task B ----------
    def task_processing():
        if state["images"] is None:
            raise RuntimeError(
                "state['images'] is None. Run Data Exploration first, or "
                "restore a session that has it."
            )

        images = pipeline.processor.ConvertImgsToRGB(state["images"])
        pipeline.processor.ReportDimensions(images)
        pipeline.processor.ExplainWhyProcessingIsRequired()

        # Stream resized+normalized batches into a pre-allocated array.
        # Peak RAM: one batch (~3 MB) + final array (~120 MB), NOT 2x.
        n = len(images)
        resized = np.empty(
            (n, *ProjectConfig.RESIZE_TO, images.shape[-1]),
            dtype=np.float32,
        )
        start = 0
        for batch in pipeline.processor.ResizeAndNormalize(
            images, batch_size=ProjectConfig.PROCESS_BATCH
        ):
            end = start + len(batch)
            resized[start:end] = batch
            start = end

        features = pipeline.processor.FlattenImages(resized)

        np.save(ProjectConfig.FEATURES_FILE, features)
        np.save(ProjectConfig.FEATURE_LABELS, state["labels"])

        state["processed_images"] = resized
        state["features"]         = features

        # Release the ~1.9 GB full-resolution buffer (3 aliases held it).
        state["images"]            = None
        pipeline.explorer.images   = None
        pipeline.processor.images  = None

        print("Reduced dimensions :", resized.shape)
        print("Feature matrix     :", features.shape)
        print("Saved features     :", ProjectConfig.FEATURES_FILE)
        print("Saved labels       :", ProjectConfig.FEATURE_LABELS)

    # ---------- Task C ----------
    def task_design():
        if state["labels"] is None:
            raise RuntimeError("state['labels'] is None. Run Data Exploration first.")

        tr, va, te = pipeline.experiment.\
            CreateStratifiedTrainingValidationAndTestSetWithAFixedRandomSeed(
                state["labels"]
            )
        pipeline.experiment.EnsureNoImageAppearsInMoreThan1Split()
        pipeline.experiment.ExplainWhyTheTestMustRemainUntouchedUntilFinalEvaluation()

        state["train_idx"] = tr
        state["val_idx"]   = va
        state["test_idx"]  = te

    # ---------- Task D ----------
    def task_training():
        required = ("features", "labels", "train_idx", "val_idx")
        missing = [k for k in required if state[k] is None]
        if missing:
            raise RuntimeError(
                f"Missing prerequisite(s): {missing}. "
                f"Complete Image Processing and Experimental Design first."
            )

        res = pipeline.trainer.CompareAtLeast3ConfigsUsingValidationPerformance(
            state["features"][state["train_idx"]],
            state["labels"][state["train_idx"]],
            state["features"][state["val_idx"]],
            state["labels"][state["val_idx"]],
        )
        pipeline.trainer.StateVariedParametersAndJustifyFinalChoice()
        GalaxyVisualizer.PlotClassifierComparison(res)

        state["training_results"] = res

    # ---------- Task E ----------
    def task_evaluation():
        required = (
            "features", "labels",
            "train_idx", "val_idx", "test_idx",
            "training_results",
        )
        missing = [k for k in required if state[k] is None]
        if missing:
            raise RuntimeError(
                f"Missing prerequisite(s): {missing}. "
                f"Complete Classifier Training first."
            )

        res = pipeline.evaluator.\
            RefitModelUsingTrainingAndValidatedDataThenEvaluateOnceOnTheTestSet(
                pipeline.trainer,
                state["features"][state["train_idx"]],
                state["labels"][state["train_idx"]],
                state["features"][state["val_idx"]],
                state["labels"][state["val_idx"]],
                state["features"][state["test_idx"]],
                state["labels"][state["test_idx"]],
            )

        # Majority-class baseline on the same test set, for a fair comparison.
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score
        baseline = DummyClassifier(strategy="most_frequent")
        train_val_idx = np.concatenate([state["train_idx"], state["val_idx"]])
        baseline.fit(state["features"][train_val_idx],
                     state["labels"][train_val_idx])
        baseline_pred = baseline.predict(state["features"][state["test_idx"]])
        baseline_acc = accuracy_score(
            state["labels"][state["test_idx"]], baseline_pred
        )

        pipeline.evaluator.CompareResultsAgainstBaseline(
            baseline_acc, res["accuracy"]
        )
        GalaxyVisualizer.PlotMetricComparison(res)

        state["evaluation_results"] = res

    # ---------- Task F ----------
    def task_interpretation():
        required = ("processed_images", "labels", "test_idx", "evaluation_results")
        missing = [k for k in required if state[k] is None]
        if missing:
            raise RuntimeError(
                f"Missing prerequisite(s): {missing}. "
                f"Complete Final Evaluation first."
            )

        pipeline.interpreter.\
            DisplayAtLeast6MisclassifiedTestImagesIfAvailableWithTrueAndPredictedLabels(
                state["processed_images"][state["test_idx"]],
                state["labels"][state["test_idx"]],
                pipeline.evaluator.predictions,
            )
        pipeline.interpreter.\
            DiscussPossibleReasonsForErrorsAndLimitationsOfFlattenedPixelsAndSimplifiedMorphologyLabels()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        terminal_header()
        terminal_status(state)
        print("\nMAIN MENU")
        print("-" * 45)
        print("1. Data Exploration")
        print("2. Image Processing")
        print("3. Experimental Design")
        print("4. Classifier Training")
        print("5. Final Evaluation")
        print("6. Error Interpretation")
        print("\n7. View Current Results")
        print("8. Save Session Now")
        print("9. Clear Saved Session")
        print("10. Reset Experiment")
        print("0. Exit")

        choice = input("\nSelect an option: ").strip()

        if   choice == "1": run_task("Data Exploration",     task_exploration)
        elif choice == "2":
            if state["images"] is None:
                print("\nRun Data Exploration first."); terminal_pause()
            else:
                run_task("Image Processing", task_processing)
        elif choice == "3":
            if state["labels"] is None:
                print("\nRun Data Exploration first."); terminal_pause()
            else:
                run_task("Experimental Design", task_design)
        elif choice == "4":
            if state["features"] is None or state["train_idx"] is None:
                print("\nComplete Image Processing and Experimental Design first."); terminal_pause()
            else:
                run_task("Classifier Training", task_training)
        elif choice == "5":
            if state["training_results"] is None:
                print("\nComplete Classifier Training first."); terminal_pause()
            else:
                run_task("Final Evaluation", task_evaluation)
        elif choice == "6":
            if state["evaluation_results"] is None:
                print("\nComplete Final Evaluation first."); terminal_pause()
            else:
                run_task("Error Interpretation", task_interpretation)
        elif choice == "7":
            terminal_header()
            print("TRAINING RESULTS:\n", state["training_results"])
            print("\nEVALUATION RESULTS:\n", state["evaluation_results"])
            terminal_pause()
        elif choice == "8":
            save_state(state, pipeline)
            terminal_pause()

        elif choice == "9":
            if SESSION_FILE.exists() and input(
                "Delete saved session? (yes/no): "
            ).strip().lower() == "yes":
                SESSION_FILE.unlink()
                print("Saved session deleted.")
            terminal_pause()
        elif choice == "10":
            if input("Reset all experiment data? (yes/no): ").strip().lower() == "yes":
                for k in state:
                    state[k] = False if isinstance(state[k], bool) else None
                print("\nExperiment state reset.")
            terminal_pause()
        elif choice == "0":
            print("\nGoodbye."); break
        else:
            print("\nInvalid selection."); terminal_pause()


if __name__ == "__main__":
    interactive_terminal()
