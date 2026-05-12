"""Tests for :class:`ImageHarmonizer` (ADR-0012)."""

from __future__ import annotations

import math

import pytest
import torch
from torchvision import tv_tensors

from segpaste._internal.gpu import harmonize as _harmonize_internal
from segpaste._internal.gpu.harmonize import HarmonizeConfig, ImageHarmonizer
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
)

_binomial_kernel_5 = _harmonize_internal._binomial_kernel_5  # pyright: ignore[reportPrivateUsage]
_dst_matrix = _harmonize_internal._dst_matrix  # pyright: ignore[reportPrivateUsage]
_dst_eigenvalues = _harmonize_internal._dst_eigenvalues  # pyright: ignore[reportPrivateUsage]

H = W = 32
B = 2
K = 2

MODES = ("reinhard", "multiband", "poisson")


def _sample(seed: int, fill: float | None = None) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    if fill is None:
        image = tv_tensors.Image(torch.rand(3, H, W, generator=gen))
    else:
        image = tv_tensors.Image(torch.full((3, H, W), fill, dtype=torch.float32))
    masks = torch.zeros(K, H, W, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
    for i in range(K):
        x1, y1 = 4 + i * 4, 4 + i * 4
        x2, y2 = x1 + 8, y1 + 8
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        labels=torch.arange(1, K + 1, dtype=torch.int64),
        instance_ids=torch.arange(K, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _padded(seed_base: int = 0, fill: float | None = None) -> PaddedBatchedDenseSample:
    samples = [_sample(seed_base + i, fill=fill) for i in range(B)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=K)


def _paste_mask() -> torch.Tensor:
    m = torch.zeros(B, H, W, dtype=torch.bool)
    m[:, 8:24, 8:24] = True
    return m


# ---------------------------------------------------------------------- #
# Identity / fast-path                                                   #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_prob_zero_returns_warped_unchanged(mode: str) -> None:
    """``prob=0`` is the graph-clean identity fast path."""
    target = _padded(seed_base=0)
    warped = _padded(seed_base=10)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode=mode, prob=0.0))  # pyright: ignore[reportArgumentType]

    out = harmonizer(target, warped, paste_mask)

    assert torch.equal(
        out.images.as_subclass(torch.Tensor),
        warped.images.as_subclass(torch.Tensor),
    )


@pytest.mark.parametrize("mode", MODES)
def test_prob_one_changes_warped(mode: str) -> None:
    """``prob=1`` runs the harmonize branch and produces a different image."""
    target = _padded(seed_base=0, fill=0.2)  # uniform grey-ish target
    warped = _padded(seed_base=10, fill=0.8)  # uniform brighter warped
    paste_mask = _paste_mask()
    gen = torch.Generator().manual_seed(0)
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode=mode, prob=1.0))  # pyright: ignore[reportArgumentType]

    out = harmonizer(target, warped, paste_mask, generator=gen)

    out_t = out.images.as_subclass(torch.Tensor)
    warped_t = warped.images.as_subclass(torch.Tensor)
    # Inside the paste region the harmonizer should pull warped toward target.
    region = out_t[:, :, 8:24, 8:24]
    assert not torch.allclose(region, warped_t[:, :, 8:24, 8:24])


# ---------------------------------------------------------------------- #
# Bernoulli mixing                                                       #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_bernoulli_per_image_mixing(mode: str) -> None:
    """``prob=0.5`` returns harmonized for some images and warped for others.

    With a seeded generator and B large enough, both branches should fire.
    """
    big_b = 32
    samples = [_sample(seed=i) for i in range(big_b)]
    warped = BatchedDenseSample.from_samples(samples).to_padded(max_instances=K)
    target = BatchedDenseSample.from_samples(
        [_sample(seed=i + 100, fill=0.5) for i in range(big_b)]
    ).to_padded(max_instances=K)
    paste_mask = torch.zeros(big_b, H, W, dtype=torch.bool)
    paste_mask[:, 8:24, 8:24] = True

    harmonizer = ImageHarmonizer(HarmonizeConfig(mode=mode, prob=0.5))  # pyright: ignore[reportArgumentType]
    gen = torch.Generator().manual_seed(0)
    out = harmonizer(target, warped, paste_mask, generator=gen)

    out_t = out.images.as_subclass(torch.Tensor)
    warped_t = warped.images.as_subclass(torch.Tensor)
    per_image_equal = torch.tensor(
        [torch.equal(out_t[i], warped_t[i]) for i in range(big_b)]
    )
    # Some images should be untouched (alpha branch), others should differ
    # (harmonize branch). Both populations non-empty.
    assert per_image_equal.any(), "expected at least one un-harmonized image"
    assert (~per_image_equal).any(), "expected at least one harmonized image"


# ---------------------------------------------------------------------- #
# Reinhard correctness                                                   #
# ---------------------------------------------------------------------- #


def test_reinhard_uniform_to_uniform_matches_target_color() -> None:
    """A uniform source patch + uniform target -> matched patch ~= target color.

    Reinhard's stat-matching collapses to identity when sigma_s = sigma_t = 0;
    we use ``min=1e-12`` so the ratio stays finite. The output mean of the
    harmonized region should land near the target background mean.
    """
    target = _padded(seed_base=0, fill=0.2)
    warped = _padded(seed_base=10, fill=0.8)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode="reinhard", prob=1.0))
    gen = torch.Generator().manual_seed(0)

    out = harmonizer(target, warped, paste_mask, generator=gen)

    out_t = out.images.as_subclass(torch.Tensor)
    region = out_t[:, :, 8:24, 8:24]
    # The harmonized region should be much closer to the target's 0.2 fill
    # than to warped's 0.8.
    region_mean = region.mean()
    assert abs(region_mean - 0.2) < abs(region_mean - 0.8)


# ---------------------------------------------------------------------- #
# Multiband: smoothness across the seam                                  #
# ---------------------------------------------------------------------- #


def test_multiband_seam_is_smoother_than_alpha() -> None:
    """Multi-band blending reduces the seam discontinuity vs. hard alpha.

    Construct uniform-color source and target with a strong color gap; the
    spatial gradient magnitude across the paste boundary should be lower for
    the multi-band blend than for a hard mask copy.
    """
    target = _padded(seed_base=0, fill=0.1)
    warped = _padded(seed_base=10, fill=0.9)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode="multiband", prob=1.0))
    gen = torch.Generator().manual_seed(0)

    out = harmonizer(target, warped, paste_mask, generator=gen)
    out_t = out.images.as_subclass(torch.Tensor)
    target_t = target.images.as_subclass(torch.Tensor)

    # Hard-alpha reference: target outside, warped inside.
    soft = paste_mask.unsqueeze(1).float()
    hard = soft * warped.images.as_subclass(torch.Tensor) + (1.0 - soft) * target_t

    # Sample a row crossing the seam at y=8 (boundary).
    row_y = 8
    seam_x_range = slice(4, 28)
    out_grad = (
        (
            out_t[:, :, row_y, seam_x_range][..., 1:]
            - out_t[:, :, row_y, seam_x_range][..., :-1]
        )
        .abs()
        .sum()
    )
    hard_grad = (
        (
            hard[:, :, row_y, seam_x_range][..., 1:]
            - hard[:, :, row_y, seam_x_range][..., :-1]
        )
        .abs()
        .sum()
    )
    assert out_grad < hard_grad


# ---------------------------------------------------------------------- #
# Poisson: relaxes outside the mask                                      #
# ---------------------------------------------------------------------- #


def test_poisson_outside_mask_close_to_target() -> None:
    """Outside the paste mask Δsrc-Δtgt = 0, so the solved field ≈ tgt."""
    target = _padded(seed_base=0, fill=0.2)
    warped = _padded(seed_base=10, fill=0.8)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode="poisson", prob=1.0))
    gen = torch.Generator().manual_seed(0)

    out = harmonizer(target, warped, paste_mask, generator=gen)
    out_t = out.images.as_subclass(torch.Tensor)
    target_t = target.images.as_subclass(torch.Tensor)

    outside_mask = ~paste_mask.unsqueeze(1)
    diff = (out_t - target_t).abs()
    # Outside the paste region, the Poisson solution should be very close
    # to the target. We tolerate small bleed from the global solve.
    assert (diff * outside_mask).max() < 0.15


# ---------------------------------------------------------------------- #
# Shape / dtype preservation                                             #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_output_shape_dtype_preserved(mode: str) -> None:
    target = _padded(seed_base=0)
    warped = _padded(seed_base=10)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode=mode, prob=1.0))  # pyright: ignore[reportArgumentType]
    gen = torch.Generator().manual_seed(0)

    out = harmonizer(target, warped, paste_mask, generator=gen)
    out_t = out.images.as_subclass(torch.Tensor)
    warped_t = warped.images.as_subclass(torch.Tensor)

    assert out_t.shape == warped_t.shape
    assert out_t.dtype == warped_t.dtype
    assert isinstance(out.images, tv_tensors.Image)
    # Other modalities pass through unchanged.
    assert torch.equal(out.boxes, warped.boxes)
    assert torch.equal(out.labels, warped.labels)
    assert torch.equal(out.instance_valid, warped.instance_valid)


@pytest.mark.parametrize("mode", MODES)
def test_output_in_unit_range(mode: str) -> None:
    """Harmonized RGB stays clamped to ``[0, 1]``."""
    target = _padded(seed_base=0)
    warped = _padded(seed_base=10)
    paste_mask = _paste_mask()
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode=mode, prob=1.0))  # pyright: ignore[reportArgumentType]
    gen = torch.Generator().manual_seed(0)

    out = harmonizer(target, warped, paste_mask, generator=gen)
    out_t = out.images.as_subclass(torch.Tensor)
    assert out_t.min() >= 0.0
    assert out_t.max() <= 1.0


# ---------------------------------------------------------------------- #
# Pyramid-depth cap                                                      #
# ---------------------------------------------------------------------- #


def test_multiband_pyramid_caps_for_small_images() -> None:
    """``pyramid_levels`` caps at ``floor(log2(min(H, W))) - 1``."""
    harmonizer = ImageHarmonizer(HarmonizeConfig(mode="multiband", pyramid_levels=10))
    # 16x16: cap = floor(log2(16)) - 1 = 3
    assert harmonizer._effective_levels(16, 16) == 3  # pyright: ignore[reportPrivateUsage]
    # 32x32: cap = floor(log2(32)) - 1 = 4; min(10, 4) = 4
    assert harmonizer._effective_levels(32, 32) == 4  # pyright: ignore[reportPrivateUsage]
    # 256x256: cap = floor(log2(256)) - 1 = 7; min(10, 7) = 7
    assert harmonizer._effective_levels(256, 256) == 7  # pyright: ignore[reportPrivateUsage]
    # User-set cap below pyramid_levels: respect the user's smaller value.
    harmonizer2 = ImageHarmonizer(HarmonizeConfig(mode="multiband", pyramid_levels=2))
    assert harmonizer2._effective_levels(256, 256) == 2  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------- #
# Math primitives                                                        #
# ---------------------------------------------------------------------- #


def test_binomial_kernel_normalized() -> None:
    k = _binomial_kernel_5()
    assert k.shape == (5, 5)
    assert math.isclose(k.sum().item(), 1.0, abs_tol=1e-6)


def test_dst_matrix_is_orthonormal() -> None:
    """The DST-I matrix is orthogonal: ``M @ M.T == I``."""
    m = _dst_matrix(8, torch.device("cpu"), torch.float64)
    eye = torch.eye(8, dtype=torch.float64)
    assert torch.allclose(m @ m.transpose(0, 1), eye, atol=1e-10)


def test_dst_eigenvalues_strictly_positive() -> None:
    eigs = _dst_eigenvalues(8, 8, torch.device("cpu"), torch.float32)
    assert eigs.shape == (8, 8)
    assert (eigs > 0).all()


# ---------------------------------------------------------------------- #
# Config validation                                                      #
# ---------------------------------------------------------------------- #


def test_config_rejects_unknown_mode() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarmonizeConfig(mode="seamlessClone")  # pyright: ignore[reportArgumentType]


def test_config_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarmonizeConfig(mode="reinhard", levels=5)  # pyright: ignore[reportCallIssue]


def test_config_rejects_prob_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarmonizeConfig(prob=1.5)
    with pytest.raises(ValidationError):
        HarmonizeConfig(prob=-0.1)
