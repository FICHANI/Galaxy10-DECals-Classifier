import numpy as np

from sklearn.model_selection import train_test_split

from config import ProjectConfig


class DesignExperiment:
    """Splitting logic; test set stays untouched until Task E."""

    def __init__(self):
        self.train_idx = None
        self.val_idx   = None
        self.test_idx  = None

    # ------------------------------------------------------------------
    def CreateStratifiedTrainingValidationAndTestSetWithAFixedRandomSeed(self, labels):
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"Expected 1-D labels; got shape {labels.shape}.")
        if len(labels) == 0:
            raise ValueError("Empty labels — cannot split.")
        if abs(ProjectConfig.TRAIN_SIZE + ProjectConfig.VALIDATION_SIZE
               + ProjectConfig.TEST_SIZE - 1.0) > 1e-6:
            raise ValueError("TRAIN/VALIDATION/TEST sizes must sum to 1.0.")

        idx = np.arange(len(labels), dtype=np.int64)

        # Step 1: train+val vs test
        idx_tv, idx_test = train_test_split(
            idx,
            test_size=ProjectConfig.TEST_SIZE,
            stratify=labels,
            random_state=ProjectConfig.RANDOM_SEED,
        )

        # Step 2: train vs val (fraction of the remaining tv pool)
        val_fraction = (
            ProjectConfig.VALIDATION_SIZE
            / (ProjectConfig.TRAIN_SIZE + ProjectConfig.VALIDATION_SIZE)
        )
        idx_train, idx_val = train_test_split(
            idx_tv,
            test_size=val_fraction,
            stratify=labels[idx_tv],
            random_state=ProjectConfig.RANDOM_SEED,
        )

        self.train_idx = np.sort(idx_train)
        self.val_idx   = np.sort(idx_val)
        self.test_idx  = np.sort(idx_test)

        print("\nSTRATIFIED SPLIT (seeded)")
        print("-" * 55)
        print(f"  train      : {len(self.train_idx):>6}  "
              f"({len(self.train_idx) / len(labels):.1%})")
        print(f"  validation : {len(self.val_idx):>6}  "
              f"({len(self.val_idx) / len(labels):.1%})")
        print(f"  test       : {len(self.test_idx):>6}  "
              f"({len(self.test_idx) / len(labels):.1%})")
        print(f"  seed       : {ProjectConfig.RANDOM_SEED}")
        print("-" * 55)

        return self.train_idx, self.val_idx, self.test_idx

    # ------------------------------------------------------------------
    def ExplainWhyTheTestMustRemainUntouchedUntilFinalEvaluation(self):
        explanation = (
            "Why the test set must remain untouched until final evaluation\n"
            "----------------------------------------------------------------\n"
            "1. Generalisation estimate — the test set is the only unbiased\n"
            "   proxy for how the model will behave on unseen galaxies. Any\n"
            "   observation of it contaminates that estimate.\n"
            "2. Information leakage — if the test labels influence which\n"
            "   classifier or hyper-parameters we pick, the reported accuracy\n"
            "   is optimistic: the model has already 'seen' the answers via\n"
            "   the selection loop.\n"
            "3. First-look rule — test performance is only valid on the FIRST\n"
            "   look. Re-running after tweaks turns the test set into a second\n"
            "   validation set, and the reported number stops meaning anything.\n"
            "4. Separation of concerns — tuning and model selection happen on\n"
            "   validation; the test set is a one-shot final exam.\n"
            "5. Reproducibility — keeping the test set sealed (seeded split,\n"
            "   no peeking) means anyone re-running the pipeline with the same\n"
            "   seed gets the same honest number."
        )
        print(explanation)
        return explanation

    # ------------------------------------------------------------------
    def EnsureNoImageAppearsInMoreThan1Split(self):
        if any(x is None for x in (self.train_idx, self.val_idx, self.test_idx)):
            raise RuntimeError(
                "Splits not set — call "
                "CreateStratifiedTrainingValidationAndTestSetWithAFixedRandomSeed() first."
            )

        tr = np.asarray(self.train_idx, dtype=np.int64)
        va = np.asarray(self.val_idx,   dtype=np.int64)
        te = np.asarray(self.test_idx,  dtype=np.int64)

        # --- 1. no duplicate index INSIDE a split ---
        for name, arr in (("train", tr), ("val", va), ("test", te)):
            if len(np.unique(arr)) != len(arr):
                raise ValueError(f"Duplicate indices inside '{name}' split.")

        # --- 2. pairwise disjoint across splits ---
        overlaps = {
            "train∩val":  np.intersect1d(tr, va),
            "train∩test": np.intersect1d(tr, te),
            "val∩test":   np.intersect1d(va, te),
        }
        for pair, shared in overlaps.items():
            if shared.size > 0:
                raise ValueError(
                    f"Split overlap detected in {pair}: "
                    f"{shared.size} shared index(es), e.g. {shared[:5].tolist()}."
                )

        # --- 3. union covers 0..N-1 exactly once (no drop) ---
        union = np.concatenate([tr, va, te])
        n_total = union.size
        expected = np.arange(n_total, dtype=np.int64)
        if not np.array_equal(np.sort(union), expected):
            missing = np.setdiff1d(expected, union)
            raise ValueError(
                f"Split coverage mismatch: {missing.size} index(es) "
                f"missing, e.g. {missing[:5].tolist()}."
            )

        # --- 4. report ---
        print("\nSPLIT INTEGRITY CHECK")
        print("-" * 55)
        print(f"  train : {len(tr):>6}")
        print(f"  val   : {len(va):>6}")
        print(f"  test  : {len(te):>6}")
        print(f"  total : {n_total:>6}  (no overlap, no drop)")
        print("  PASSED")
        print("-" * 55)

        return True