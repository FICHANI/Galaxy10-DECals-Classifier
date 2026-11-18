# src/exploration.py
"""
Task A — Data exploration.
Responsible for loading Galaxy10, reporting the raw class distribution,
and producing a balanced, memory-bounded subset for downstream tasks.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from config import ProjectConfig


class ExploreData:
    """Loads Galaxy10_DECals.h5 and inspects it."""

    def __init__(self, data_path=None):
        self.data_path       = data_path or ProjectConfig.DATA_PATH
        self.images          = None            # (N, 256, 256, 3) uint8 — subset
        self.labels          = None            # (N,) int
        self.original_labels = None            # full-dataset labels (reporting)

    # ------------------------------------------------------------------
    def LoadDataset(self):
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)

        with h5py.File(self.data_path, "r") as f:
            if ProjectConfig.HDF5_KEY_LABELS not in f:
                raise KeyError(
                    f"'{ProjectConfig.HDF5_KEY_LABELS}' missing. "
                    f"Keys present: {list(f.keys())}"
                )
            if ProjectConfig.HDF5_KEY_IMAGES not in f:
                raise KeyError(
                    f"'{ProjectConfig.HDF5_KEY_IMAGES}' missing. "
                    f"Keys present: {list(f.keys())}"
                )

            # --- 1. labels are small — read everything for the class report ---
            self.original_labels = f[ProjectConfig.HDF5_KEY_LABELS][:]

            # --- 2. pick the balanced subset BEFORE touching the images ---
            # Deterministic via RANDOM_SEED so runs are reproducible.
            rng = np.random.default_rng(ProjectConfig.RANDOM_SEED)
            selected = []

            for class_id in range(ProjectConfig.NUM_CLASSES):
                class_idx = np.where(self.original_labels == class_id)[0]
                rng.shuffle(class_idx)
                selected.extend(class_idx[:ProjectConfig.MAX_SAMPLES_PER_CLASS])

            selected = np.asarray(selected, dtype=np.int64)
            rng.shuffle(selected)               # one global shuffle, once
            order         = np.argsort(selected)   # sorted(selected) = selected[order]
            sorted_idx    = selected[order]        # non-decreasing → h5py-safe
            inverse_order = np.argsort(order)      # maps sorted buffer → shuffled order

            # --- 3. now stream ONLY the selected rows from HDF5 ---
            images_dset = f[ProjectConfig.HDF5_KEY_IMAGES]
            n     = len(sorted_idx)
            chunk = ProjectConfig.HDF5_CHUNK

            raw = np.empty(
                (n, *images_dset.shape[1:]),
                dtype=images_dset.dtype,
            )
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                raw[start:end] = images_dset[sorted_idx[start:end]]   # increasing → OK

            # labels aligned to the same order as self.images
            #self.images = np.empty_like(raw)
            #self.images[inverse_order] = raw      # in-place scatter, no 2× peak

            self.images = raw[inverse_order]
            del raw

            self.labels = self.original_labels[selected].astype(np.int64)
            
        # --- report ---
        print(f"Full dataset size    : {len(self.original_labels)}")
        print(f"Balanced subset size : {n}")
        print(f"Loaded images shape  : {self.images.shape}  ({self.images.dtype})")
        print(f"Loaded labels shape  : {self.labels.shape}")

        return self.images, self.labels

    # ------------------------------------------------------------------
    def ReportOriginalClassDistribution(self):
        if self.original_labels is None:
            raise RuntimeError(
                "original_labels is empty — call LoadDataset() first."
            )

        # --- 1. count occurrences per class id ---
        counts = Counter(self.original_labels)
        total = len(self.original_labels)

        # --- 2. build a table ordered by class id, with human-readable names ---
        rows = []
        for class_id in range(ProjectConfig.NUM_CLASSES):
            count = counts.get(class_id, 0)
            rows.append({
                "Class":       class_id,
                "Class Name":  ProjectConfig.CLASS_NAMES[class_id],
                "Count":       count,
                "Percentage":  round(100.0 * count / total, 2) if total else 0.0,
            })

        report = pd.DataFrame(rows)

        # --- 3. print it ---
        print("\nORIGINAL CLASS DISTRIBUTION (full dataset, before balancing)")
        print("-" * 65)
        print(report.to_string(index=False))
        print("-" * 65)
        print(f"Total samples: {total}")

        # --- 4. persist as CSV ---
        out_path = ProjectConfig.REPORTS_PATH / "original_class_distribution.csv"
        report.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        return report

    # ------------------------------------------------------------------
    def ApplyPrescribedClassMapping(self):
        if self.labels is None:
            raise RuntimeError("labels is empty — call LoadDataset() first.")

        expected = np.arange(ProjectConfig.NUM_CLASSES, dtype=np.int64)
        present  = np.unique(self.labels)

        # --- 1. every label must be a valid class id in [0, NUM_CLASSES) ---
        unknown = np.setdiff1d(present, expected)
        if unknown.size > 0:
            raise ValueError(
                f"Unexpected label(s) {unknown.tolist()} found. "
                f"Expected integers in [0, {ProjectConfig.NUM_CLASSES - 1}]. "
                f"CLASS_NAMES has {len(ProjectConfig.CLASS_NAMES)} entries."
            )

        # --- 2. every class name must have a slot (sanity on config) ---
        if len(ProjectConfig.CLASS_NAMES) != ProjectConfig.NUM_CLASSES:
            raise ValueError(
                f"Config mismatch: NUM_CLASSES={ProjectConfig.NUM_CLASSES} but "
                f"CLASS_NAMES has {len(ProjectConfig.CLASS_NAMES)} entries."
            )

        # --- 3. warn (don't fail) if the balanced subset dropped a class ---
        #     The full dataset always has all 10 classes; a subset only
        #     misses one if MAX_SAMPLES_PER_CLASS was set to 0 for a rare class.
        missing = np.setdiff1d(expected, present)
        if missing.size > 0:
            missing_names = [ProjectConfig.CLASS_NAMES[i] for i in missing]
            print(
                f"[WARN] Balanced subset is missing class id(s) {missing.tolist()} "
                f"({missing_names}). Stratified splitting will fail — "
                f"raise MAX_SAMPLES_PER_CLASS in config.py."
            )

        # --- 4. cast to int64 for sklearn; confirm we didn't lose range ---
        before = self.labels.copy()
        self.labels = self.labels.astype(np.int64)

        if not np.array_equal(before.astype(np.int64), self.labels):
            raise ValueError("Label cast to int64 altered values — aborting.")

        # --- 5. print the mapping we just confirmed ---
        print("\nCLASS MAPPING (integer id → morphology label)")
        print("-" * 55)
        for cid in range(ProjectConfig.NUM_CLASSES):
            marker = "✓" if cid in present else "✗ (absent)"
            print(f"  {cid}  →  {ProjectConfig.CLASS_NAMES[cid]:<28} {marker}")
        print("-" * 55)
        print(f"Total classes present in subset: {present.size}/{ProjectConfig.NUM_CLASSES}")

        return self.labels

    # ------------------------------------------------------------------
    def SelectABalancedSubset(self):
        if self.images is None or self.labels is None:
            raise RuntimeError("Call LoadDataset() first.")

        cap     = ProjectConfig.MAX_SAMPLES_PER_CLASS
        n_class = ProjectConfig.NUM_CLASSES

        # ------------------------------------------------------------------
        # FAST PATH — already balanced? Return without copying anything.
        # ------------------------------------------------------------------
        counts = np.bincount(self.labels, minlength=n_class)
        if np.all(counts <= cap):
            indices = np.arange(len(self.labels), dtype=np.int64)
            print(
                f"Balanced subset already in memory "
                f"({len(self.labels)} samples, cap={cap}/class, no copy)."
            )
            print(f"Per-class counts: {counts.tolist()}")
            return self.images, self.labels, indices

        # ------------------------------------------------------------------
        # SLOW PATH — reduce to the prescribed cap.
        # ------------------------------------------------------------------
        rng = np.random.default_rng(ProjectConfig.RANDOM_SEED)
        selected_parts = []

        for class_id in range(n_class):
            idx = np.where(self.labels == class_id)[0]
            if idx.size == 0:
                print(
                    f"[WARN] Class {class_id} "
                    f"({ProjectConfig.CLASS_NAMES[class_id]}) has 0 samples "
                    f"— cannot select {cap}."
                )
                continue
            rng.shuffle(idx)
            selected_parts.append(idx[:cap])

        if not selected_parts:
            raise RuntimeError("No class contains any samples.")

        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)                 # one global shuffle


        n          = len(selected)
        new_images = np.empty(
            (n, *self.images.shape[1:]), dtype=self.images.dtype
        )
        chunk = ProjectConfig.HDF5_CHUNK
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            new_images[start:end] = self.images[selected[start:end]]

        self.images = new_images               # old array released by GC
        self.labels = self.labels[selected]    # labels are small — cheap copy

        print(f"Reduced subset to {n} samples (cap={cap}/class).")
        print(f"Balanced dataset: {self.images.shape}")
        return self.images, self.labels, selected

    # ------------------------------------------------------------------
    def DisplayAtLeast12RepresentativeImagesAndDescribeVisibleDifferences(self):
        if self.images is None or self.labels is None:
            raise RuntimeError("Call LoadDataset() first.")

        rng = np.random.default_rng(ProjectConfig.RANDOM_SEED)
        n   = len(self.labels)

        # ------------------------------------------------------------------
        # 1. pick one representative index per class (seeded, reproducible)
        # ------------------------------------------------------------------
        def pick_index(class_id, exclude=None):
            idx = np.where(self.labels == class_id)[0]
            if exclude is not None:
                idx = idx[idx != exclude]
            if idx.size == 0:
                return None
            return int(idx[rng.integers(idx.size)])

        picks = []                                  # (class_id, dataset_index, tag)
        for class_id in range(ProjectConfig.NUM_CLASSES):
            i = pick_index(class_id)
            if i is not None:
                picks.append((class_id, i, "representative"))

        # ------------------------------------------------------------------
        # 2. top-up to 12 with extra samples of the confusable pair
        # ------------------------------------------------------------------
        extras = [
            (2, "extra — confusable pair with class 3"),
            (3, "extra — confusable pair with class 2"),
        ]
        for class_id, tag in extras:
            prior = next((p[1] for p in picks if p[0] == class_id), None)
            i = pick_index(class_id, exclude=prior)
            if i is not None:
                picks.append((class_id, i, tag))

        # If a class is empty in the subset (very low MAX_SAMPLES_PER_CLASS),
        # keep topping up from classes that do exist so we still reach 12.
        while len(picks) < 12:
            avail = np.unique(self.labels)
            if avail.size == 0:
                break
            class_id = int(avail[rng.integers(avail.size)])
            prior = {p[1] for p in picks if p[0] == class_id}
            idx = np.setdiff1d(np.where(self.labels == class_id)[0],
                              np.array(list(prior), dtype=np.int64))
            if idx.size == 0:
                break
            picks.append((class_id, int(idx[rng.integers(idx.size)]), "top-up"))

        picks = picks[:12]                          # never exceed the grid

        # ------------------------------------------------------------------
        # 3. render the 3x4 grid
        # ------------------------------------------------------------------
        def displayable(img):
            """imshow-safe copy: uint8 passthrough; float [0,1] or [0,255] handled."""
            if img.dtype == np.uint8:
                return img
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() <= 1.0001:
                return np.clip(arr, 0.0, 1.0)
            return np.clip(arr / 255.0, 0.0, 1.0)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flat

        for cell, ax in enumerate(axes):
            if cell >= len(picks):
                ax.axis("off")
                continue
            class_id, dataset_idx, tag = picks[cell]
            ax.imshow(displayable(self.images[dataset_idx]))
            ax.set_title(
                f"class {class_id}: {ProjectConfig.CLASS_NAMES[class_id]}\n"
                f"[{tag}, idx={dataset_idx}]",
                fontsize=10,
            )
            ax.axis("off")

        fig.suptitle(
            "Galaxy10 DECals — Representative Morphologies (g/r/z composite)",
            fontsize=14, y=0.995,
        )
        plt.tight_layout()

        out_path = ProjectConfig.FIGURES_PATH / "representative_galaxies.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out_path}")

        # ------------------------------------------------------------------
        # 4. describe the visible differences
        # ------------------------------------------------------------------
        per_class_notes = {
            0: "Irregular, asymmetric, no coherent arms — often warped or "
              "tidally distorted; sometimes clumpy.",
            1: "Two or more distinct nuclei, tidal bridges/tails, "
              "asymmetric silhouette.",
            2: "Circular/elliptical, smooth brightness profile, "
              "centrally concentrated, no visible arms.",
            3: "Roughly round but with faint structure or slight ellipticity; "
              "boundary with class 2 is subjective — hence the extra samples.",
            4: "Elongated, smooth, no arms; axis ratio > ~0.5 with no "
              "edge-on signature.",
            5: "Central bar with trailing spiral arms emerging from the bar ends.",
            6: "Tightly wound arms, small pitch angle, compact "
              "two-arm structure.",
            7: "Loose, open, flocculent arms; large pitch angle.",
            8: "Thin edge-on disk, no visible central bulge; "
              "flat light profile.",
            9: "Edge-on disk WITH a distinct central bulge — a bright "
              "round core embedded in the disk.",
        }

        print("\nVISIBLE DIFFERENCES BETWEEN MORPHOLOGY CLASSES")
        print("-" * 65)
        for class_id in range(ProjectConfig.NUM_CLASSES):
            print(f"{class_id} — {ProjectConfig.CLASS_NAMES[class_id]:<28} "
                  f"{per_class_notes.get(class_id, '')}")

        print("\nKEY OBSERVATIONS FOR THE REPORT")
        print("-" * 65)
        print(
            "• The dominant visual cues are: (i) presence/absence of spiral "
            "structure, (ii) central bulge prominence, (iii) axis ratio "
            "(round vs edge-on), (iv) symmetry.\n"
            "• Classes 2 and 3 are visually adjacent in a 32×32 view — arm and "
            "bulge detail collapses below pixel resolution, so a pixel-based "
            "model is expected to confuse them. The two extra slots above were "
            "picked to make this boundary explicit.\n"
            "• Classes 8 and 9 differ only by bulge brightness — colour "
            "gradient across the central few pixels.\n"
            "• Disturbed (0) and Merging (1) both break symmetry, but Merging "
            "adds multiple nuclei/tails, which Disturbed need not have.\n"
            "• Colour (g/r/z) carries real signal here, but it is lost if "
            "you flatten to grayscale — the classifier below keeps all three "
            "channels for this reason."
        )

        return picks