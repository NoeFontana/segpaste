"""PyTorch transforms for copy-paste augmentation."""

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.types import BatchedDenseSample, DenseSample


class CopyPasteCollator:
    """Collate function for batching data with copy-paste augmentation.

    Consumes ``list[DenseSample]`` and produces a :class:`BatchedDenseSample`.
    Every sample in the batch acts as a source for every other sample; target
    exclusion is performed by index rather than image-identity comparison.
    """

    def __init__(self, augmentation: CopyPasteAugmentation) -> None:
        """Initialize copy-paste collator.

        Args:
            augmentation: Copy-paste augmentation instance.
        """
        self.copy_paste: CopyPasteAugmentation = augmentation

    def __call__(self, batch: list[DenseSample]) -> BatchedDenseSample:
        """Apply copy-paste augmentation across the batch and stack the result.

        For each sample ``i``, the remaining samples ``[j for j != i]`` act as
        source objects. Empty batches return an empty :class:`BatchedDenseSample`
        (``batch_size == 0``).
        """
        if not batch:
            return BatchedDenseSample.from_samples([])

        augmented: list[DenseSample] = [
            self.copy_paste.transform(
                target, [s for j, s in enumerate(batch) if j != i]
            )
            for i, target in enumerate(batch)
        ]
        return BatchedDenseSample.from_samples(augmented)
