"""Parity gate: ``forward`` and ``forward_with_audit`` agree (ADR-0014 PR3).

The audit-returning sibling method must not perturb the training output.
A seeded generator + a fresh generator state for each call gives both
codepaths the same RNG draw order; every output tensor must match bitwise.
"""

from __future__ import annotations

import torch

from scripts.compile_explain import build_fixture
from segpaste import BatchAuditPacket, BatchCopyPaste


def _equal_optional(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    assert (a is None) == (b is None), "presence mismatch"
    if a is None or b is None:
        return True
    return bool(torch.equal(a, b))


def test_forward_with_audit_first_element_matches_forward() -> None:
    """``forward(...) == forward_with_audit(...)[0]`` bitwise."""
    padded = build_fixture()
    module = BatchCopyPaste()

    gen_a = torch.Generator().manual_seed(0xC0FFEE)
    gen_b = torch.Generator().manual_seed(0xC0FFEE)
    out_forward = module.forward(padded, generator=gen_a)
    out_audit, _ = module.forward_with_audit(padded, generator=gen_b)

    assert torch.equal(
        out_forward.images.as_subclass(torch.Tensor),
        out_audit.images.as_subclass(torch.Tensor),
    )
    assert torch.equal(out_forward.boxes, out_audit.boxes)
    assert torch.equal(out_forward.labels, out_audit.labels)
    assert torch.equal(out_forward.instance_valid, out_audit.instance_valid)
    assert _equal_optional(out_forward.instance_masks, out_audit.instance_masks)
    assert _equal_optional(out_forward.instance_ids, out_audit.instance_ids)
    if out_forward.semantic_maps is not None:
        assert out_audit.semantic_maps is not None
        assert torch.equal(
            out_forward.semantic_maps.as_subclass(torch.Tensor),
            out_audit.semantic_maps.as_subclass(torch.Tensor),
        )
    assert _equal_optional(out_forward.depth, out_audit.depth)
    assert _equal_optional(out_forward.depth_valid, out_audit.depth_valid)
    assert _equal_optional(out_forward.normals, out_audit.normals)


def test_forward_with_audit_returns_named_tuple() -> None:
    """``forward_with_audit`` returns ``(sample, BatchAuditPacket)``."""
    padded = build_fixture()
    module = BatchCopyPaste()
    result = module.forward_with_audit(padded)
    assert isinstance(result, tuple)
    assert len(result) == 2
    _, audit = result
    assert isinstance(audit, BatchAuditPacket)


def test_audit_packet_paste_union_shape() -> None:
    """``audit.paste_union`` is ``[B, H, W]`` bool."""
    padded = build_fixture()
    module = BatchCopyPaste()
    _, audit = module.forward_with_audit(padded)
    b = padded.batch_size
    h, w = padded.images.shape[-2:]
    assert audit.paste_union.shape == (b, h, w)
    assert audit.paste_union.dtype == torch.bool


def test_audit_packet_thresholds_mirror_config() -> None:
    """``audit.thresholds`` mirrors the module's config fractional knobs."""
    padded = build_fixture()
    module = BatchCopyPaste()
    _, audit = module.forward_with_audit(padded)
    assert (
        audit.thresholds.min_residual_area_frac == module.config.min_residual_area_frac
    )
    assert audit.thresholds.tau_stuff_frac is None  # no panoptic config in default
