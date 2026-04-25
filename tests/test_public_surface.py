"""Public-surface enforcement tests.

The public API of ``segpaste`` is pinned by ADR-0001 Part (i) (as amended by
ADR-0003). These tests are the forcing function that prevents surface
re-accretion during P1 churn: adding, removing, or renaming a public name
requires amending both ``segpaste.__all__`` and the ``_EXPECTED_PUBLIC_API``
constant below, which in turn forces an ADR amendment.
"""

from __future__ import annotations

import inspect

import segpaste

# Pinned public surface. Any drift fails this test. Amendments require an
# ADR update; see ADR-0001 Part (i) and ADR-0003.
_EXPECTED_PUBLIC_API: tuple[str, ...] = (
    "BatchCopyPaste",
    "BatchedDenseSample",
    "CameraIntrinsics",
    "CocoDetectionV2",
    "DenseSample",
    "FixedSizeCrop",
    "InstanceMask",
    "Modality",
    "PaddedBatchedDenseSample",
    "PaddingMask",
    "PanopticMap",
    "PanopticSchema",
    "PresetConfig",
    "RandomResize",
    "SanitizeBoundingBoxes",
    "SemanticMap",
    "__version__",
    "create_coco_dataloader",
    "get_preset",
    "list_presets",
    "make_large_scale_jittering",
    "register_preset",
)


def test_all_matches_pinned_surface() -> None:
    """``segpaste.__all__`` matches the ADR-pinned public surface."""
    assert tuple(sorted(segpaste.__all__)) == tuple(sorted(_EXPECTED_PUBLIC_API))


def test_every_all_entry_resolves() -> None:
    """Every name in ``__all__`` is actually defined on the module."""
    missing = [name for name in segpaste.__all__ if not hasattr(segpaste, name)]
    assert not missing, f"names in __all__ not defined on segpaste: {missing}"


def test_no_non_all_public_attrs() -> None:
    """Only ``__all__`` entries and submodules are visible as public attrs.

    Dunders and ``_``-prefixed names are allowed through; imported submodules
    (``segpaste.augmentation`` etc.) are allowed through because Python
    registers them as attributes implicitly.
    """
    allowed: set[str] = set(segpaste.__all__)
    leaks: list[str] = []
    for name in dir(segpaste):
        if name.startswith("_"):
            continue
        if name in allowed:
            continue
        attr = getattr(segpaste, name)
        if inspect.ismodule(attr):
            continue
        leaks.append(name)
    assert not leaks, f"public attrs not in __all__: {leaks}"


def test_version_attribute_is_string() -> None:
    """``segpaste.__version__`` is a non-empty string."""
    assert isinstance(segpaste.__version__, str)
    assert segpaste.__version__


def test_copy_paste_transform_is_gone() -> None:
    """ADR-0003: ``CopyPasteTransform`` is deleted outright in 0.9.0."""
    assert not hasattr(segpaste, "CopyPasteTransform")
    from segpaste import augmentation

    assert not hasattr(augmentation, "CopyPasteTransform")


def test_detection_target_not_public() -> None:
    """ADR-0003: ``DetectionTarget`` is no longer a top-level export."""
    assert "DetectionTarget" not in segpaste.__all__
    assert not hasattr(segpaste, "DetectionTarget")


def test_detection_target_removed_from_types_by_0_9_1() -> None:
    """ADR-0003: ``DetectionTarget`` is fully deleted in 0.9.1."""
    from segpaste import types

    assert not hasattr(types, "DetectionTarget")
