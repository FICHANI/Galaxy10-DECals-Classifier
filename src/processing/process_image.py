import numpy as np

from skimage.transform import resize

from config import ProjectConfig


class ProcessImage:
    """Consistent pixel-based preprocessing for training and inference."""

    def __init__(self):
        self.images = None

    # ------------------------------------------------------------------
    def ConvertImgsToRGB(self, images):
        images = np.asarray(images)

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected images with shape (N, H, W, 3); got {images.shape}."
            )
        if images.shape[0] == 0:
            raise ValueError("Empty image array — nothing to process.")

        # Explicitly refuse a silent float upcast here. Float conversion
        # happens AFTER resize in NormalizeImages(), when the array is tiny.
        if images.dtype != np.uint8:
            print(
                f"[INFO] ConvertImgsToRGB received dtype={images.dtype}; "
                f"expected uint8. Passing through unchanged."
            )

        self.images = images        # reference, not a copy
        return self.images

    # ------------------------------------------------------------------
    def ReportDimensions(self, images=None):
        images = self.images if images is None else images

        if images is None:
            raise ValueError("No images supplied and self.images is unset.")

        images = np.asarray(images)

        if images.size == 0:
            raise ValueError("Empty image array — nothing to report.")

        # compute min/max once on the reduced axes (cheap, no copy)
        lo = float(images.min())
        hi = float(images.max())

        print("\nIMAGE DIMENSIONS")
        print("-" * 55)
        print(f"  shape    : {images.shape}")
        print(f"  ndim     : {images.ndim}")
        print(f"  dtype    : {images.dtype}")
        print(f"  min      : {lo}")
        print(f"  max      : {hi}")
        print(f"  size     : {images.size} values "
              f"({images.nbytes / 1e6:.1f} MB in memory)")

        if images.dtype == np.uint8:
            print("  range OK : 8-bit intensities (0–255)")
        elif hi <= 1.0001 and lo >= 0.0:
            print("  range OK : normalised float in [0, 1]")
        else:
            print("  range ?? : unexpected — check preprocessing chain")

        print("-" * 55)
        return images.shape

    # ------------------------------------------------------------------

    def ExplainWhyProcessingIsRequired(self):
        exp = (
            "1. Images must have consistent dimensions.\n"
            "2. Pixel values must be numeric and ML-friendly.\n"
            "3. g/r/z channels retain their scientific meaning.\n"
            "4. Flattened arrays must have a fixed feature count.\n"
            "5. Preprocessing must be identical at train and infer time."
        )
        explanation = (
            "Why image processing is required for Galaxy10 DECals\n"
            "-----------------------------------------------------\n"
            "1. Consistent shape — DECals tiles are already (256, 256, 3) "
            "uint8, but downstream tasks need a uniform shape, so we resize "
            "to 32x32. Unresized input would give 196 608 features per image; "
            "resized gives 3 072 — a ~64x reduction that is tractable for "
            "CPU-only classical ML.\n"
            "2. Numeric dtype — raw tiles are uint8; sklearn estimators "
            "expect float32/float64. Upcast is deferred until AFTER resize "
            "so the float copy is small, not a ~7.8 GB float32 clone of the "
            "full-resolution set.\n"
            "3. Preserved astronomical channels — the three channels are "
            "g/r/z, not RGB. We keep them in their stored order; no colour "
            "remapping is applied, because the physics encoded in each band "
            "must survive preprocessing.\n"
            "4. Fixed feature count — flattening (32, 32, 3) → (3072,) "
            "gives every sample the same row width, which is a hard "
            "requirement for tabular classifiers and for train/test matrix "
            "alignment.\n"
            "5. Train/infer parity — the SAME resize, same /255 normalisation, "
            "and same flatten order must be applied to any new image at "
            "inference. Any drift here silently shifts the feature "
            "distribution and destroys generalisation."
        )
        print(exp)
        print(f"\n{explanation}")
        return explanation

    # ------------------------------------------------------------------
    def ResizeAndNormalize(self, images, batch_size=ProjectConfig.PROCESS_BATCH):
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected (N, H, W, 3); got {images.shape}."
            )
        n = len(images)
        if n == 0:
            return
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images[start:end]
            resized = np.stack([
                resize(im, ProjectConfig.RESIZE_TO,
                       preserve_range=True, anti_aliasing=True)
                for im in batch
            ]).astype(np.float32)
            resized /= ProjectConfig.NORMALIZE_SCALE
            yield resized

    # ------------------------------------------------------------------
    '''def FlattenImages(self, images):
        """(N, 32, 32, 3) → (N, 3072) float32 pixel features."""
        images = np.asarray(images)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / ProjectConfig.NORMALIZE_SCALE
        else:
            images = images.astype(np.float32, copy=False)
        return images.reshape(len(images), -1)'''

    def FlattenImages(self, images):
        """
        (N, 32, 32, 3) → (N, 3072) float32.

        Input may be:
        - uint8 from the raw HDF5 read (or an early pipeline stage) → normalise once
        - float32 from ResizeAndNormalize, already in [0, 1]          → pass through

        The dtype check prevents double-normalisation: dividing a float
        that is already in [0, 1] by 255 would collapse the features to
        ~0.004 and destroy all signal.
        """
        images = np.asarray(images)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / ProjectConfig.NORMALIZE_SCALE
        else:
            images = images.astype(np.float32, copy=False)
        return images.reshape(len(images), -1)