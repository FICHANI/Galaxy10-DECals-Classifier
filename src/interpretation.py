import numpy as np
import matplotlib.pyplot as plt

from config import ProjectConfig


class InterpretErrors:
    """Diagnostics; relies on EvaluateOnUnseenImgs.predictions."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    def DisplayAtLeast6MisclassifiedTestImagesIfAvailableWithTrueAndPredictedLabels(
        self, images, y_true, y_pred, max_show=6
    ):
        images = np.asarray(images)
        y_true = np.asarray(y_true, dtype=np.int64, copy=False)
        y_pred = np.asarray(y_pred, dtype=np.int64, copy=False)

        if len(images) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: images={len(images)}, "
                f"y_true={len(y_true)}, y_pred={len(y_pred)}."
            )

        # ------------------------------------------------------------------
        # 1. locate misclassifications
        # ------------------------------------------------------------------
        wrong = np.where(y_true != y_pred)[0]

        if wrong.size == 0:
            print("\nNo misclassifications found in the test set. "
                  "Nothing to display.")
            return

        n_show = min(max_show, wrong.size)
        picks  = wrong[:n_show]

        print(f"\nMISCLASSIFIED TEST EXAMPLES")
        print("-" * 62)
        print(f"  misclassified : {wrong.size} / {len(y_true)} "
              f"({wrong.size / len(y_true):.1%})")
        print(f"  showing       : {n_show}")

        # ------------------------------------------------------------------
        # 2. figure: 2 rows x 3 cols, scaled if max_show > 6
        # ------------------------------------------------------------------
        cols = min(3, n_show)
        rows = int(np.ceil(n_show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_2d(axes).ravel()

        def displayable(img):
            """imshow-safe copy: uint8 passthrough; float [0,1] or [0,255]."""
            if img.dtype == np.uint8:
                return img
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() <= 1.0001:
                return np.clip(arr, 0.0, 1.0)
            return np.clip(arr / 255.0, 0.0, 1.0)

        for cell, test_pos in enumerate(picks):
            ax = axes[cell]
            ax.imshow(displayable(images[test_pos]))
            true_id = int(y_true[test_pos])
            pred_id = int(y_pred[test_pos])
            ax.set_title(
                f"true: {ProjectConfig.CLASS_NAMES[true_id]} ({true_id})\n"
                f"pred: {ProjectConfig.CLASS_NAMES[pred_id]} ({pred_id})",
                fontsize=9,
            )
            ax.axis("off")
            print(
                f"  [{cell + 1}] test_pos={test_pos}  "
                f"true={true_id} ({ProjectConfig.CLASS_NAMES[true_id]})  →  "
                f"pred={pred_id} ({ProjectConfig.CLASS_NAMES[pred_id]})"
            )

        # blank any unused cells
        for cell in range(n_show, len(axes)):
            axes[cell].axis("off")

        fig.suptitle(
            f"Misclassified test galaxies ({n_show} of {wrong.size})",
            fontsize=12, y=0.995,
        )
        plt.tight_layout()

        out = ProjectConfig.FIGURES_PATH / "misclassified_examples.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

        # ------------------------------------------------------------------
        # 3. most common confusions (feeds the discussion text)
        # ------------------------------------------------------------------
        pairs = {}
        for i in wrong:
            key = (int(y_true[i]), int(y_pred[i]))
            pairs[key] = pairs.get(key, 0) + 1
        top = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:5]

        print("\nTOP CONFUSION PAIRS (true → predicted)")
        print("-" * 62)
        for (t, p), count in top:
            print(
                f"  {t} {ProjectConfig.CLASS_NAMES[t]:<28} → "
                f"{p} {ProjectConfig.CLASS_NAMES[p]:<28}  ×{count}"
            )
        print("-" * 62)

    # ------------------------------------------------------------------
    def DiscussPossibleReasonsForErrorsAndLimitationsOfFlattenedPixelsAndSimplifiedMorphologyLabels(
        self,
    ):
        discussion = (
            "POSSIBLE REASONS FOR ERRORS AND LIMITATIONS\n"
            "-------------------------------------------------------------\n"
            "1. Flattened pixels discard spatial structure.\n"
            "   Each 32x32x3 tile becomes a 3072-D vector in a fixed\n"
            "   raster order. Rotating the same spiral by 90 degrees\n"
            "   produces a very different vector even though it is the\n"
            "   same galaxy. Classical classifiers (LinearSVC, logistic\n"
            "   regression) have no built-in rotation or translation\n"
            "   invariance, so orientation dominates the feature space\n"
            "   and small shifts push samples across decision boundaries.\n"
            "\n"
            "2. Morphology labels are coarse and soft.\n"
            "   The Galaxy10 labels come from the DECals morphological\n"
            "   catalogue, which is a human-annotated bucketing. Boundaries\n"
            "   between 'Round Smooth' (2) and 'In-between Round Smooth' (3),\n"
            "   or between 'Edge-on without Bulge' (8) and 'Edge-on with\n"
            "   Bulge' (9), are subjective. At 32x32 most of the visual cue\n"
            "   that distinguishes them (bulge brightness, subtle ellipticity)\n"
            "   has already been averaged away, so the classifier is being\n"
            "   asked to separate classes that even experts disagree on.\n"
            "\n"
            "3. Colour matters, but the resolution kills it.\n"
            "   The three stored g/r/z channels carry real astrophysical\n"
            "   signal (star formation, dust, redshift). At full resolution\n"
            "   that signal is concentrated in the arms and central regions.\n"
            "   Downsampling to 32x32 with anti-aliasing averages those\n"
            "   regions together, washing out the subtle colour gradients\n"
            "   that separate barred from unbarred spirals, or tight from\n"
            "   loose arms.\n"
            "\n"
            "4. Class imbalance in the raw dataset.\n"
            "   Before balancing, Galaxy10 is dominated by 'Round Smooth'\n"
            "   (~15%) and 'Cigar Shaped Smooth' (~14%), while 'Disturbed'\n"
            "   is ~4%. Even after per-class capping (MAX_SAMPLES_PER_CLASS),\n"
            "   the minority classes have fewer examples to learn from, so\n"
            "   the classifier's decision boundary is pulled toward the\n"
            "   majority classes. This shows up as recall < precision on\n"
            "   the small classes in the per-class metrics table.\n"
            "\n"
            "5. What a CNN on full resolution would capture.\n"
            "   Convolutional networks preserve locality: filters respond to\n"
            "   local patterns (arm segments, bars, bulge gradients) and\n"
            "   pooling layers build translation invariance. At 256x256 with\n"
            "   three channels, a small CNN can learn features that a 3072-D\n"
            "   flat vector simply cannot represent. The price is compute:\n"
            "   this pipeline is CPU-only, so the classical-ML results are\n"
            "   the honest trade-off between accuracy and tractability.\n"
            "\n"
            "6. Additional limitations of this specific pipeline.\n"
            "   - Fixed random seed gives ONE split; different seeds would\n"
            "     shift the test accuracy by a couple of points.\n"
            "   - Only one chosen classifier is evaluated on the test set.\n"
            "     A seed-averaged ensemble would give a tighter estimate.\n"
            "   - 32x32 resize is a choice, not a proven optimum. Higher\n"
            "     resolution increases feature count quadratically and\n"
            "     linear-SVM training time linearly in that feature count,\n"
            "     which would exceed CPU-only memory budgets.\n"
            "   - The majority-class baseline confirms the model learns\n"
            "     something, but does not establish that the model has\n"
            "     learnt the *right* thing. A rotation-invariant baseline\n"
            "     (e.g. rotation-averaged features) would be a stronger\n"
            "     sanity check.\n"
        )
        print(discussion)
        return discussion